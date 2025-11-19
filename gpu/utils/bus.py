from amaranth_soc.wishbone.bus import Signature

internal_gpu_bus = Signature(
    addr_width=32,
    data_width=32,
    granularity=8,
)
