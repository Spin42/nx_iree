#include "runtime.h"

#include <iree/async/frontier_tracker.h>
#include <iree/async/util/proactor_pool.h>
#include <iree/base/threading/numa.h>
#include <iree/hal/api.h>
#include <iree/hal/drivers/init.h>
#include <iree/modules/hal/debugging.h>
#include <iree/vm/bytecode/module.h>
#include <iree/tooling/device_util.h>

#ifdef DEBUG
#include <iree/base/tracing/tracy.h>
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef CUDA_ENABLED
#include <iree/hal/drivers/cuda/cuda_device.h>
#define RUN_IF_CUDA_ENABLED(CODE) CODE
#else
#define RUN_IF_CUDA_ENABLED(CODE) ;
#endif

#define RETURN_PAIR_IF_ERROR(status)                                           \
  if (!iree_status_is_ok(status)) {                                            \
    return {status, std::nullopt};                                             \
  }

iree::runtime::Device::~Device() {
  if (ref) {
    iree_hal_device_release(ref);
  }
}

iree::runtime::IREETensor::IREETensor(iree_hal_buffer_view_t *buffer_view,
                                      iree_hal_element_type_t type,
                                      iree_hal_device_t *device,
                                      bool copy_buffer) {
  this->buffer_view = buffer_view;
  this->type = type;
  this->device = device;
  size = iree_hal_buffer_view_byte_length(this->buffer_view);

  iree_host_size_t shape_rank =
      iree_hal_buffer_view_shape_rank(this->buffer_view);
  const iree_hal_dim_t *hal_dims =
      iree_hal_buffer_view_shape_dims(this->buffer_view);

  dims = std::vector<iree_hal_dim_t>();
  dims.reserve(shape_rank);
  for (int i = 0; i < shape_rank; i++) {
    dims.push_back(hal_dims[i]);
  }

  data = nullptr;
}

iree::runtime::IREETensor::IREETensor(void *data, size_t size,
                                      std::vector<int64_t> in_dims,
                                      iree_hal_element_type_t type)
    : size(size), type(type) {
  dims.reserve(in_dims.size());

  for (auto dim : in_dims) {
    dims.push_back(static_cast<iree_hal_dim_t>(dim));
  }

  this->data = std::malloc(size); // Allocate memory
  std::memcpy(this->data, data, size);

  this->buffer_view = nullptr;
}

iree::runtime::IREETensor::IREETensor(char *buffer) {
  size_t offset = 0;

  // Deserialize 'type'
  std::memcpy(&type, buffer + offset, sizeof(type));
  offset += sizeof(type);

  // Deserialize 'size'
  std::memcpy(&size, buffer + offset, sizeof(size));
  offset += sizeof(size);

  // Allocate memory and deserialize 'data'
  data = operator new(size); // Allocate raw memory
  std::memcpy(data, buffer + offset, size);
  offset += size;

  // Deserialize 'dims'
  size_t num_dims;
  std::memcpy(&num_dims, buffer + offset, sizeof(num_dims));
  offset += sizeof(num_dims);
  dims.resize(num_dims);
  std::memcpy(dims.data(), buffer + offset, num_dims * sizeof(iree_hal_dim_t));

  this->buffer_view = nullptr;
}

iree::runtime::IREETensor::~IREETensor() { this->deallocate(); }

void iree::runtime::IREETensor::deallocate() {
  if (data != nullptr) {
    std::free(data);
    data = nullptr;
  }

  if (buffer_view != nullptr) {
    iree_hal_buffer_view_release(buffer_view);
    buffer_view = nullptr;
  }
}

iree_status_t
iree::runtime::IREETensor::upload(iree_hal_device_t *target_device) {
  if (buffer_view != nullptr) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      target_device, iree_hal_device_allocator(target_device), dims.size(),
      dims.data(), type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      data_byte_span(), &buffer_view));

  device = target_device;
  std::free(data);
  data = nullptr;
  return iree_ok_status();
}

std::vector<char> *iree::runtime::IREETensor::serialize() {
  auto buffer = new std::vector<char>();

  // Serialize 'type'
  size_t type_size = sizeof(type);
  buffer->insert(buffer->end(), reinterpret_cast<const char *>(&type),
                 reinterpret_cast<const char *>(&type) + type_size);

  // Serialize 'size'
  size_t size_size = sizeof(size);
  buffer->insert(buffer->end(), reinterpret_cast<const char *>(&size),
                 reinterpret_cast<const char *>(&size) + size_size);

  if (data == nullptr) {
    data = std::malloc(size);

    if (data == nullptr) {
      return nullptr;
    }

    auto status = read_buffer(device, buffer_view, data, size);

    if (!iree_status_is_ok(status)) {
      return nullptr;
    }
  }
  // Serialize 'data'
  buffer->insert(buffer->end(), reinterpret_cast<const char *>(data),
                 reinterpret_cast<const char *>(data) + size);

  // Serialize 'dims'
  size_t dims_size = sizeof(iree_hal_dim_t) * dims.size();
  size_t num_dims = dims.size();
  buffer->insert(buffer->end(), reinterpret_cast<const char *>(&num_dims),
                 reinterpret_cast<const char *>(&num_dims) + sizeof(num_dims));
  buffer->insert(buffer->end(), reinterpret_cast<const char *>(dims.data()),
                 reinterpret_cast<const char *>(dims.data()) + dims_size);

  return buffer;
}

iree_hal_element_type_t nx_type_to_iree_type(std::string type) {
  using type_enum = iree_hal_element_types_t;

  if (type == "s8") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_INT_8;
  } else if (type == "s16") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_INT_16;
  } else if (type == "s32") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_INT_32;
  } else if (type == "s64") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_INT_64;
  } else if (type == "u8") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_UINT_8;
  } else if (type == "u16") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_UINT_16;
  } else if (type == "u32") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_UINT_32;
  } else if (type == "u64") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_UINT_64;
  } else if (type == "bf16") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
  } else if (type == "f16") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  } else if (type == "f32") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  } else if (type == "f64") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_64;
  } else if (type == "c64") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
  } else if (type == "c128") {
    return type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
  }

  std::cout << "Unknown type" << std::endl;
  return type_enum::IREE_HAL_ELEMENT_TYPE_NONE;
}

iree_vm_instance_t *create_instance() {
  iree_vm_instance_t *instance = nullptr;
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance);
  if (!iree_status_is_ok(status)) {
    return nullptr;
  }

  status = iree_hal_module_register_all_types(instance);
  if (!iree_status_is_ok(status)) {
    iree_vm_instance_release(instance);
    return nullptr;
  }

  return instance;
}

// Devices need a proactor pool for async I/O; one pool serves every device in
// the process and lives as long as the process.
static iree_status_t
device_create_params(iree_hal_device_create_params_t *out_params) {
  static std::mutex pool_mutex;
  static iree_async_proactor_pool_t *pool = nullptr;

  std::lock_guard<std::mutex> guard(pool_mutex);
  if (pool == nullptr) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &pool));
  }

  *out_params = iree_hal_device_create_params_default();
  out_params->proactor_pool = pool;
  return iree_ok_status();
}

// A device's queues join exactly one frontier tracker, so every module loaded
// on a device shares one device group. Groups live as long as the process.
static iree_status_t shared_device_group(iree_hal_device_t *device,
                                         iree_hal_device_group_t **out_group) {
  static std::mutex groups_mutex;
  static std::map<iree_hal_device_t *, iree_hal_device_group_t *> groups;

  std::lock_guard<std::mutex> guard(groups_mutex);
  auto it = groups.find(device);
  if (it != groups.end()) {
    *out_group = it->second;
    return iree_ok_status();
  }

  iree_async_frontier_tracker_t *frontier_tracker = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_t *group = nullptr;
  iree_status_t status = iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &group);
  iree_async_frontier_tracker_release(frontier_tracker);
  IREE_RETURN_IF_ERROR(status);

  groups[device] = group;
  *out_group = group;
  return iree_ok_status();
}

iree_status_t register_all_drivers(iree_hal_driver_registry_t *registry) {
  iree_status_t status = iree_hal_register_all_available_drivers(registry);
  return status;
}

std::pair<iree_status_t, std::vector<iree::runtime::Driver *>>
list_drivers(iree_hal_driver_registry_t *registry) {
  size_t driver_info_count;
  iree_hal_driver_info_t *driver_infos;

  iree_status_t status = iree_hal_driver_registry_enumerate(
      registry, iree_allocator_system(), &driver_info_count, &driver_infos);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), driver_infos);
    return {status, {}};
  }

  std::vector<iree::runtime::Driver *> drivers;

  for (size_t i = 0; i < driver_info_count; i++) {
    auto info = driver_infos[i];

    auto driver =
        new iree::runtime::Driver(info.driver_name.data, info.driver_name.size,
                                  info.full_name.data, info.full_name.size);

    if (driver->name == "hip") {
      continue;
    }
    drivers.push_back(driver);
  }

  iree_allocator_free(iree_allocator_system(), driver_infos);
  return {iree_ok_status(), drivers};
}

iree_status_t list_devices(iree_hal_driver_registry_t *registry,
                           std::vector<iree::runtime::Device *> &devices) {
  auto [status, drivers] = list_drivers(registry);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  for (auto driver : drivers) {
    std::vector<iree::runtime::Device *> driver_devices;

    status = list_devices(registry, driver->name, driver_devices);
    if (!iree_status_is_ok(status)) {
      for (auto device : devices) {
        delete device;
      }
      devices.clear();
      return status;
    }

    devices.insert(devices.end(), driver_devices.begin(), driver_devices.end());
  }

  return iree_ok_status();
}

iree_status_t list_devices(iree_hal_driver_registry_t *registry,
                           std::string driver_name,
                           std::vector<iree::runtime::Device *> &devices) {
  size_t device_info_count;
  iree_hal_device_info_t *device_infos;
  iree_hal_driver_t *driver;

  iree_status_t status = iree_hal_driver_registry_try_create(
      registry, iree_make_cstring_view(driver_name.c_str()),
      iree_allocator_system(), &driver);

  if (!iree_status_is_ok(status)) {
    iree_hal_driver_release(driver);
    return status;
  }

  status = iree_hal_driver_query_available_devices(
      driver, iree_allocator_system(), &device_info_count, &device_infos);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), device_infos);
    iree_hal_driver_release(driver);
    return status;
  }

  for (size_t i = 0; i < device_info_count; i++) {
    auto device = new iree::runtime::Device(driver_name);
    auto info = device_infos[i];
    std::string device_urn(info.path.data, info.path.size);
    device->uri = driver_name + "://" + device_urn;
    device->id = info.device_id;

    iree_hal_device_create_params_t create_params;
    status = device_create_params(&create_params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_create_device(
          registry, iree_make_cstring_view(device->uri.c_str()), &create_params,
          iree_allocator_system(), &device->ref);
    }

    // Queues accept work only once the device belongs to a group
    iree_hal_device_group_t *group = nullptr;
    if (iree_status_is_ok(status)) {
      status = shared_device_group(device->ref, &group);
    }

    if (!iree_status_is_ok(status)) {
      delete device;
      for (auto created : devices) {
        delete created;
      }
      devices.clear();
      iree_hal_driver_release(driver);
      iree_allocator_free(iree_allocator_system(), device_infos);
      return status;
    }

    RUN_IF_CUDA_ENABLED(if (driver_name == "cuda") {
      const iree_hal_cuda_dynamic_symbols_t *cuda_symbols =
          iree_hal_cuda_device_dynamic_symbols(device->ref);
      auto ctx = iree_hal_cuda_device_context(device->ref);
      cuda_symbols->cuCtxSetCurrent(ctx);
    });

    devices.push_back(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
  iree_hal_driver_release(driver);
  return iree_ok_status();
}

iree_hal_device_t *create_device(iree_hal_driver_registry_t *registry,
                                 const std::string &device_uri) {
  iree_hal_device_t *device = nullptr;

  iree_hal_device_create_params_t create_params;
  iree_status_t status = device_create_params(&create_params);
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        registry, iree_make_cstring_view(device_uri.c_str()), &create_params,
        iree_allocator_system(), &device);
  }

  // Queues accept work only once the device belongs to a group
  iree_hal_device_group_t *group = nullptr;
  if (iree_status_is_ok(status)) {
    status = shared_device_group(device, &group);
  }

  RUN_IF_CUDA_ENABLED(if (device_uri.find("cuda://") != std::string::npos) {
    const iree_hal_cuda_dynamic_symbols_t *cuda_symbols =
        iree_hal_cuda_device_dynamic_symbols(device);
    auto ctx = iree_hal_cuda_device_context(device);
    cuda_symbols->cuCtxSetCurrent(ctx);
  });

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release(device);
    return nullptr;
  }

  return device;
}

iree::runtime::LoadedModule::~LoadedModule() {
  if (context != nullptr) {
    iree_vm_context_release(context);
    context = nullptr;
  }
  if (device != nullptr) {
    iree_hal_device_release(device);
    device = nullptr;
  }
  // The bytecode module references the archive without owning it, so the
  // archive is freed only after the context has released the module.
  if (archive != nullptr) {
    std::free(archive);
    archive = nullptr;
  }
}

std::pair<iree_status_t, iree::runtime::LoadedModule *>
load_module(iree_vm_instance_t *instance, iree_hal_device_t *device,
            std::string driver_name, unsigned char *bytecode,
            size_t bytecode_size) {
  const char kMainFunctionName[] = "module.main";
  auto loaded = new iree::runtime::LoadedModule();

  RUN_IF_CUDA_ENABLED(if (driver_name == "cuda") {
    const iree_hal_cuda_dynamic_symbols_t *cuda_symbols =
        iree_hal_cuda_device_dynamic_symbols(device);
    auto ctx = iree_hal_cuda_device_context(device);
    cuda_symbols->cuCtxSetCurrent(ctx);
  });

  // FlatBuffer archives are read in place and need 64-byte alignment.
  size_t archive_size = (bytecode_size + 63) & ~static_cast<size_t>(63);
  loaded->archive = std::aligned_alloc(64, archive_size);
  if (loaded->archive == nullptr) {
    delete loaded;
    return {iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "unable to allocate the module archive"),
            nullptr};
  }
  std::memcpy(loaded->archive, bytecode, bytecode_size);

  iree_hal_device_retain(device);
  loaded->device = device;

  iree_hal_device_group_t *device_group = nullptr;
  iree_vm_module_t *hal_module = nullptr;
  iree_vm_module_t *bytecode_module = nullptr;

  iree_status_t status = shared_device_group(device, &device_group);

  if (iree_status_is_ok(status)) {
    status = iree_hal_module_create(
        instance, iree_hal_module_device_policy_default(), device_group,
        IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
        iree_hal_module_debug_sink_null(), iree_allocator_system(),
        &hal_module);
  }

  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_make_const_byte_span(loaded->archive, bytecode_size),
        iree_allocator_null(), iree_allocator_system(), &bytecode_module);
  }

  if (iree_status_is_ok(status)) {
    iree_vm_module_t *modules[] = {hal_module, bytecode_module};
    status = iree_vm_context_create_with_modules(
        instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules),
        &modules[0], iree_allocator_system(), &loaded->context);
  }

  // The context retains both modules.
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  if (iree_status_is_ok(status)) {
    status = iree_vm_context_resolve_function(
        loaded->context, iree_make_cstring_view(kMainFunctionName),
        &loaded->main_function);
  }

  if (iree_status_is_ok(status)) {
    iree_vm_function_signature_t signature =
        iree_vm_function_signature(&loaded->main_function);
    iree_string_view_t input_signature;
    iree_string_view_t output_signature;
    status = iree_vm_function_call_get_cconv_fragments(
        &signature, &input_signature, &output_signature);
    loaded->output_count = output_signature.size;
  }

  if (!iree_status_is_ok(status)) {
    delete loaded;
    return {status, nullptr};
  }

  return {iree_ok_status(), loaded};
}

std::pair<iree_status_t,
          std::optional<std::vector<iree::runtime::IREETensor *>>>
call_loaded(iree::runtime::LoadedModule *loaded, std::string driver_name,
            std::vector<iree::runtime::IREETensor *> exla_inputs) {
  iree_hal_device_t *device = loaded->device;
  iree_vm_list_t *inputs = nullptr;
  iree_vm_list_t *outputs = nullptr;

  RUN_IF_CUDA_ENABLED(if (driver_name == "cuda") {
    const iree_hal_cuda_dynamic_symbols_t *cuda_symbols =
        iree_hal_cuda_device_dynamic_symbols(device);
    auto ctx = iree_hal_cuda_device_context(device);
    cuda_symbols->cuCtxSetCurrent(ctx);
  });

  iree_status_t status = iree_vm_list_create(
      iree_vm_make_undefined_type_def(), exla_inputs.size(),
      iree_allocator_system(), &inputs);

  for (auto input : exla_inputs) {
    if (!iree_status_is_ok(status)) break;

    if (input->buffer_view == nullptr && input->data == nullptr) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "input tensor has been deallocated");
      break;
    }

    iree_vm_ref_t arg_buffer_view_ref;

    if (input->buffer_view) {
      // The tensor keeps its buffer view, so the list takes a new reference.
      arg_buffer_view_ref = iree_hal_buffer_view_retain_ref(input->buffer_view);
    } else {
      iree_hal_buffer_view_t *arg_buffer_view = nullptr;
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, iree_hal_device_allocator(device), input->dims.size(),
          input->dims.data(), input->type,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          },
          input->data_byte_span(), &arg_buffer_view);
      if (!iree_status_is_ok(status)) break;

      arg_buffer_view_ref = iree_hal_buffer_view_move_ref(arg_buffer_view);
    }
    status = iree_vm_list_push_ref_move(inputs, &arg_buffer_view_ref);
  }

  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                 loaded->output_count, iree_allocator_system(),
                                 &outputs);
  }

  if (iree_status_is_ok(status)) {
    // A VM context runs one invocation at a time.
    std::lock_guard<std::mutex> guard(loaded->mutex);
    status = iree_vm_invoke(loaded->context, loaded->main_function,
                            IREE_VM_INVOCATION_FLAG_NONE,
                            /*policy=*/NULL, inputs, outputs,
                            iree_allocator_system());
  }

  std::vector<iree::runtime::IREETensor *> results;
  if (iree_status_is_ok(status)) {
    iree_host_size_t output_count = iree_vm_list_size(outputs);
    results.reserve(output_count);
    for (iree_host_size_t i = 0; i < output_count; i++) {
      iree_hal_buffer_view_t *output_buffer_view =
          iree_vm_list_get_buffer_view_retain(outputs, i);
      if (!output_buffer_view) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "can't get output buffer view [index=%d]",
                                  (int)i);
        break;
      }

      iree_hal_element_type_t out_type =
          iree_hal_buffer_view_element_type(output_buffer_view);
      results.push_back(
          new iree::runtime::IREETensor(output_buffer_view, out_type, device));
    }
  }

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);

  if (!iree_status_is_ok(status)) {
    for (auto tensor : results) delete tensor;
    return {status, std::nullopt};
  }
  return {iree_ok_status(), results};
}

std::pair<iree_status_t,
          std::optional<std::vector<iree::runtime::IREETensor *>>>
call(iree_vm_instance_t *instance, iree_hal_device_t *device,
     std::string driver_name, unsigned char *bytecode, size_t bytecode_size,
     std::vector<iree::runtime::IREETensor *> exla_inputs) {
  auto [status, loaded] =
      load_module(instance, device, driver_name, bytecode, bytecode_size);
  if (!iree_status_is_ok(status)) {
    return {status, std::nullopt};
  }
  auto result = call_loaded(loaded, driver_name, exla_inputs);
  delete loaded;
  return result;
}

iree_status_t read_buffer(iree_hal_device_t *device,
                          iree_hal_buffer_view_t *buffer_view,
                          void *output_buffer, size_t num_bytes) {
  if (!buffer_view) {
    return iree_make_status(IREE_STATUS_OK);
  }

  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(buffer_view);

  iree_device_size_t num_bytes_actual =
      num_bytes == -1 ? iree_hal_buffer_byte_length(buffer)
                      : (iree_device_size_t)num_bytes;

  iree_string_view_t device_id = iree_hal_device_id(device);

  std::string device_id_str = std::string(device_id.data, device_id.size);

  RUN_IF_CUDA_ENABLED(if (device_id_str.find("cuda") != std::string::npos) {
    const iree_hal_cuda_dynamic_symbols_t *cuda_symbols =
        iree_hal_cuda_device_dynamic_symbols(device);
    auto ctx = iree_hal_cuda_device_context(device);
    cuda_symbols->cuCtxSetCurrent(ctx);
  });

  iree_status_t status = iree_hal_device_transfer_d2h(
      device, buffer, 0, output_buffer, num_bytes_actual,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());

  return status;
}

std::string get_status_message(iree_status_t status) {
  char *status_string = NULL;
  size_t status_length = 0;

  auto system_allocator = iree_allocator_system();

  iree_status_to_string(status, &system_allocator, &status_string,
                        &status_length);

  std::stringstream ss;
  ss << "Failed to execute IREE runtime due to error: ";
  ss << status_string;
  iree_status_free(status);
  return ss.str();
}

bool is_ok(iree_status_t status) { return iree_status_is_ok(status); }

iree_hal_driver_registry_t *get_driver_registry() {
  return iree_hal_available_driver_registry();
}