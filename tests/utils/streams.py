from typing import Callable

from amaranth import *
from amaranth.lib import stream
from amaranth.sim import Simulator, SimulatorContext


async def stream_get(
    ctx: SimulatorContext, stream: stream.Interface, finish: Value
) -> list:
    results = []
    while not ctx.get(finish):
        await ctx.tick().until(stream.valid | finish)

        while ctx.get(stream.valid):
            results.append(ctx.get(stream.payload))
            ctx.set(stream.ready, 1)
            await ctx.tick()
            ctx.set(stream.ready, 0)

    return results


async def stream_put(
    ctx: SimulatorContext, stream: stream.Interface, data: list
) -> None:
    for item in data:
        ctx.set(stream.payload, item)
        ctx.set(stream.valid, 1)
        await ctx.tick().until(stream.ready)
        ctx.set(stream.valid, 0)


def stream_testbench(
    sim: Simulator,
    input_stream: stream.Interface | None = None,
    input_data: list | None = None,
    output_stream: stream.Interface | None = None,
    expected_output_data: list | None = None,
    is_finished: Value = C(1),
    init_process: Callable | None = None,
) -> None:
    is_initialized = Signal(init=0)
    all_data_sent = Signal(init=0)

    async def input_tb(ctx: SimulatorContext):
        await ctx.tick().until(is_initialized)
        if input_stream is not None:
            await stream_put(ctx, input_stream, input_data)
        ctx.set(all_data_sent, 1)

    async def output_tb(ctx: SimulatorContext):
        await ctx.tick().until(is_initialized)
        if output_stream is not None:
            results = await stream_get(ctx, output_stream, all_data_sent & is_finished)
            print("Output data:", results)
            assert ctx.get(
                results == expected_output_data
            ), "Output data does not match expected data"

    async def init_tb(ctx: SimulatorContext):
        if init_process is not None:
            await init_process(ctx)
        ctx.set(is_initialized, 1)

    sim.add_testbench(input_tb)
    sim.add_testbench(output_tb)
    sim.add_testbench(init_tb)
