import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.utils import fixed
from gpu.utils.math import FixedPointVecNormalize
from gpu.utils.types import FixedPoint, Vector3


@pytest.mark.parametrize(
    "data, expected",
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([3.0, 4.0, 0.0], [0.6, 0.8, 0.0]),
        ([-3.0, -4.0, 0.0], [-0.6, -0.8, 0.0]),
    ]
    + [([float(i), 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(1, 10)],
)
def test_normalize(data: list[float], expected: list[float]):
    rst = Signal(1)
    dut = ResetInserter(rst)(FixedPointVecNormalize(Vector3))

    async def tb_operation(data, expected, ctx):
        ctx.set(rst, 1)
        ctx.set(dut.start, 0)
        await ctx.tick()
        ctx.set(rst, 0)
        await ctx.tick()

        print()
        print(f"Testing with input: {data}")

        ctx.set(dut.value, [fixed.Const(v, FixedPoint) for v in data])
        ctx.set(dut.start, 1)

        await ctx.tick()
        ctx.set(dut.start, 0)

        print("Waiting for ready signal...")

        cycles = 0
        while not ctx.get(dut.ready):
            await ctx.tick()
            cycles += 1

        result = ctx.get(dut.result)
        result = [ctx.get(r).as_float() for r in result]
        print(f"got {result}; expected {expected} ({cycles=})")

        await ctx.tick()

        if not all(abs(r - e) < 1e-1 for r, e in zip(result, expected)):
            raise AssertionError("Test failed!")

    async def tb_normalize(ctx):
        await tb_operation(data, expected, ctx)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb_normalize)

    try:
        sim.run()
    except Exception:
        sim.reset()
        with sim.write_vcd(
            "test_fixed_point_vec_normalize.vcd",
            "test_fixed_point_vec_normalize.gtkw",
            traces=dut,
        ):
            sim.run()
