import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.input_assembly.cores import InputAssembly
from gpu.input_assembly.layouts import InputData, InputMode
from gpu.utils import fixed
from gpu.utils.types import FixedPoint, FixedPoint_mem, Vector3, Vector4, Vector4_mem

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench

vec0001_mem = Vector4_mem.const(
    [
        fixed.Const(0.0, FixedPoint_mem),
        fixed.Const(0.0, FixedPoint_mem),
        fixed.Const(0.0, FixedPoint_mem),
        fixed.Const(1.0, FixedPoint_mem),
    ]
)

vec0001 = Vector4.const(
    [
        fixed.Const(0.0, FixedPoint),
        fixed.Const(0.0, FixedPoint),
        fixed.Const(0.0, FixedPoint),
        fixed.Const(1.0, FixedPoint),
    ]
)

vec000 = Vector3.const(
    [
        fixed.Const(0.0, FixedPoint),
        fixed.Const(0.0, FixedPoint),
        fixed.Const(0.0, FixedPoint),
    ]
)


default_data = InputData.const({"constant_value": vec0001_mem})


def make_test_input_assembly(
    test_name: str,
    addr: int,
    input_idx: list[int],
    memory_data: bytes,
    expected: list,
    pos_mode: InputMode = InputMode.CONSTANT,
    pos_data: InputData = default_data,
    norm_mode: InputMode = InputMode.CONSTANT,
    norm_data: InputData = default_data,
    tex0_mode: InputMode = InputMode.CONSTANT,
    tex0_data: InputData = default_data,
    tex1_mode: InputMode = InputMode.CONSTANT,
    tex1_data: InputData = default_data,
    color_mode: InputMode = InputMode.CONSTANT,
    color_data: InputData = default_data,
):
    t = SimpleTestbench(InputAssembly(), mem_addr=addr, mem_size=1024)

    t.set_csrs(
        t.dut.csr_bus,
        [
            ((("position", "mode"),), pos_mode.value.to_bytes(1, "little")),
            ((("position", "data"),), pos_data.as_bits().to_bytes(16, "little")),
            ((("normal", "mode"),), norm_mode.value.to_bytes(1, "little")),
            ((("normal", "data"),), norm_data.as_bits().to_bytes(16, "little")),
            ((("texcoords", "0", "mode"),), tex0_mode.value.to_bytes(1, "little")),
            ((("texcoords", "0", "data"),), tex0_data.as_bits().to_bytes(16, "little")),
            ((("texcoords", "1", "mode"),), tex1_mode.value.to_bytes(1, "little")),
            ((("texcoords", "1", "data"),), tex1_data.as_bits().to_bytes(16, "little")),
            ((("color", "mode"),), color_mode.value.to_bytes(1, "little")),
            ((("color", "data"),), color_data.as_bits().to_bytes(16, "little")),
        ],
        "input_assembly",
    )

    t.arbiter.add(t.dut.bus)
    t.make()

    async def tb(ctx):
        await t.initialize_memory(ctx, addr, memory_data)
        await t.initialize_csrs(ctx)

    sim = Simulator(t.m)
    sim.add_clock(1e-9)
    stream_testbench(
        sim,
        init_process=tb,
        input_stream=t.dut.is_index,
        input_data=input_idx,
        output_stream=t.dut.os_vertex,
        expected_output_data=expected,
        is_finished=t.dut.ready,
    )

    try:
        sim.run()
    except Exception:
        sim.reset()

        with sim.write_vcd(f"{test_name}.vcd", f"{test_name}.gtkw", traces=t.dut):
            sim.run()


vec1234_mem = Vector4_mem.const(
    [
        fixed.Const(1.0, FixedPoint_mem),
        fixed.Const(2.0, FixedPoint_mem),
        fixed.Const(3.0, FixedPoint_mem),
        fixed.Const(4.0, FixedPoint_mem),
    ]
)

vec1234 = Vector4.const(
    [
        fixed.Const(1.0, FixedPoint),
        fixed.Const(2.0, FixedPoint),
        fixed.Const(3.0, FixedPoint),
        fixed.Const(4.0, FixedPoint),
    ]
)

vec5678_mem = Vector4_mem.const(
    [
        fixed.Const(5.0, FixedPoint_mem),
        fixed.Const(6.0, FixedPoint_mem),
        fixed.Const(7.0, FixedPoint_mem),
        fixed.Const(8.0, FixedPoint_mem),
    ]
)

vec5678 = Vector4.const(
    [
        fixed.Const(5.0, FixedPoint),
        fixed.Const(6.0, FixedPoint),
        fixed.Const(7.0, FixedPoint),
        fixed.Const(8.0, FixedPoint),
    ]
)


def test_input_assembly_constant_only():
    make_test_input_assembly(
        test_name="test_input_assembly_constant_only",
        addr=0x80000000,
        memory_data=b"",
        input_idx=[0, 1, 2, 3, 4],
        expected=[
            {
                "position": vec0001,
                "normal": vec000,
                "texcoords": [vec0001, vec0001],
                "color": vec0001,
            }
            for _ in range(5)
        ],
    )


@pytest.mark.parametrize(
    ["test_name", "comp", "comp_in", "separation"],
    [
        ("test_input_assembly_continous_pos", "position", "pos", 0),
        ("test_input_assembly_continous_norm", "normal", "norm", 0),
        ("test_input_assembly_continous_tex0", "texcoords[0]", "tex0", 0),
        ("test_input_assembly_continous_tex1", "texcoords[1]", "tex1", 0),
        ("test_input_assembly_continous_col", "color", "color", 0),
        ("test_input_assembly_strided_4_pos", "position", "pos", 4),
        ("test_input_assembly_strided_8_norm", "normal", "norm", 8),
        ("test_input_assembly_strided_12_tex0", "texcoords[0]", "tex0", 12),
        ("test_input_assembly_strided_16_tex1", "texcoords[1]", "tex1", 16),
        ("test_input_assembly_strided_20_col", "color", "color", 20),
    ],
)
def test_input_assembly_single_component(test_name, comp, comp_in, separation):
    expected = [
        {
            "position": vec0001,
            "normal": vec000,
            "texcoords": [vec0001, vec0001],
            "color": vec0001,
        },
        {
            "position": vec0001,
            "normal": vec000,
            "texcoords": [vec0001, vec0001],
            "color": vec0001,
        },
    ]

    if comp == "texcoords[0]":
        expected[0]["texcoords"][0] = vec1234
        expected[1]["texcoords"][0] = vec5678
    elif comp == "texcoords[1]":
        expected[0]["texcoords"][1] = vec1234
        expected[1]["texcoords"][1] = vec5678
    elif comp == "normal":
        expected[0][comp] = Vector3.const(vec1234[:3])
        expected[1][comp] = Vector3.const(vec5678[:3])
    else:
        expected[0][comp] = vec1234
        expected[1][comp] = vec5678

    v = {
        f"{comp_in}_mode": InputMode.PER_VERTEX,
        f"{comp_in}_data": InputData.const(
            {
                "per_vertex": {
                    "address": 0x80000000,
                    "stride": (Vector4_mem.size // 8) + separation,
                }
            }
        ),
    }

    make_test_input_assembly(
        test_name=test_name,
        addr=0x80000000,
        memory_data=b"".join(
            (
                vec1234_mem.as_bits().to_bytes(16, "little"),
                b"\x00" * separation,
                vec5678_mem.as_bits().to_bytes(16, "little"),
            )
        ),
        input_idx=[0, 1],
        expected=expected,
        **v,
    )
