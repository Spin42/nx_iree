defmodule NxIREE.Device do
  @moduledoc false

  # Devices are created on first use rather than at application start: driver
  # setup can be slow or fail (e.g. a GPU that did not come up), and on an
  # embedded system that should not take the application down at boot.
  #
  # Configuration:
  #
  #   * `config :nx_iree, drivers: ["vulkan", "local-sync"]` - only probe
  #     these drivers (default: all registered drivers)
  #   * `config :nx_iree, default_device: "vulkan://..."` - the device used
  #     when none is given (default: the first GPU, else the first CPU device)
  #   * `config :nx_iree, software_vulkan: true` - also open software Vulkan
  #     devices such as lavapipe (default: false)

  @state_key {__MODULE__, :state}

  defstruct [:ref, :driver_name, :kind, :id, :uri, :compiler_target_backend]

  @doc """
  Creates the devices if that has not happened yet.
  """
  def init do
    case :persistent_term.get(@state_key, nil) do
      nil -> :global.trans({{__MODULE__, :init}, self()}, &init_once/0, [node()])
      _state -> :ok
    end
  end

  defp init_once do
    case :persistent_term.get(@state_key, nil) do
      nil -> :persistent_term.put(@state_key, create_devices())
      _state -> :ok
    end
  end

  defp state do
    :ok = init()
    :persistent_term.get(@state_key)
  end

  defp create_devices do
    {:ok, registry} = NxIREE.Native.get_driver_registry()
    {:ok, drivers} = NxIREE.Native.list_drivers(registry)

    drivers =
      Map.new(drivers, fn {name, full_name} -> {to_string(name), to_string(full_name)} end)

    enabled =
      case Application.get_env(:nx_iree, :drivers, :all) do
        :all -> drivers
        names -> Map.take(drivers, Enum.map(names, &to_string/1))
      end

    software? = Application.get_env(:nx_iree, :software_vulkan, false)

    devices =
      enabled
      |> Map.keys()
      |> Enum.flat_map(fn name ->
        # A driver whose system library is missing (e.g. no libcuda) has no
        # devices
        case NxIREE.Native.list_devices(registry, name, software?) do
          {:ok, devices} -> devices
          {:error, _} -> []
        end
      end)
      |> Enum.map(fn {device_ref, driver_name, device_uri, device_id} ->
        driver_name = to_string(driver_name)

        %__MODULE__{
          uri: to_string(device_uri),
          ref: device_ref,
          driver_name: driver_name,
          kind: if(List.starts_with?(device_uri, ~c"local-sync"), do: :cpu, else: :io),
          id: device_id,
          compiler_target_backend: compiler_target_backend(driver_name)
        }
      end)
      |> sort_devices_by_priority()

    %{
      registry: registry,
      drivers: enabled,
      devices: devices,
      default: configured_default(devices) || pick_default(enabled, devices)
    }
  end

  defp configured_default(devices) do
    case Application.get_env(:nx_iree, :default_device) do
      nil ->
        nil

      uri ->
        Enum.find(devices, &(&1.uri == uri)) ||
          raise ArgumentError,
                "configured :default_device #{inspect(uri)} is not among the available " <>
                  "devices: #{inspect(Enum.map(devices, & &1.uri))}"
    end
  end

  defp pick_default(drivers, devices) do
    drivers
    |> Map.keys()
    |> Enum.sort_by(&if(&1 in ["cuda", "rocm", "metal", "vulkan"], do: 0, else: 1))
    |> Enum.find_value(fn driver -> Enum.find(devices, &(&1.driver_name == driver)) end)
  end

  defp compiler_target_backend("metal"), do: "metal-spirv"
  defp compiler_target_backend("cuda"), do: "cuda"
  defp compiler_target_backend("rocm"), do: "rocm"
  defp compiler_target_backend("vulkan"), do: "vulkan-spirv"
  defp compiler_target_backend("local-sync"), do: "llvm-cpu"
  defp compiler_target_backend("local-task"), do: "llvm-cpu"
  defp compiler_target_backend(_), do: nil

  defp sort_devices_by_priority(devices) do
    Enum.sort_by(devices, fn device ->
      if String.ends_with?(device.uri, "://") or String.ends_with?(device.uri, "://default") do
        -1
      else
        device.id
      end
    end)
  end

  def list_drivers do
    {:ok, state().drivers}
  end

  def list do
    {:ok, state().devices}
  end

  def list(driver) do
    driver = to_string(driver)
    {:ok, Enum.filter(state().devices, &(&1.driver_name == driver))}
  end

  def get(nil) do
    {:ok, default_device()}
  end

  def get(%__MODULE__{} = device) do
    {:ok, device}
  end

  def get(device_uri) do
    case Enum.find(state().devices, &(&1.uri == device_uri)) do
      nil -> {:error, :unknown_device}
      device -> {:ok, device}
    end
  end

  @doc """
  Like `get/1`, raising when the device does not exist.
  """
  def fetch!(device) do
    case get(device) do
      {:ok, nil} ->
        raise ArgumentError, "no IREE device is available"

      {:ok, device} ->
        device

      {:error, :unknown_device} ->
        raise ArgumentError,
              "unknown IREE device #{inspect(device)}, available: " <>
                inspect(Enum.map(state().devices, & &1.uri))
    end
  end

  def default_device do
    state().default
  end

  def find_default_device do
    default_device()
  end

  def find_default_device(driver) do
    {:ok, devices} = list(driver)
    Enum.at(devices, 0)
  end
end
