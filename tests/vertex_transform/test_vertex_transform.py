import numpy as np
import pytest
from amaranth import *
from amaranth.sim import Simulator
from numpy.linalg import inv

from gpu.utils.types import FixedPoint_mem
from gpu.vertex_transform.cores import VertexTransform

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def identity_mat(size):
    return np.identity(size)


def make_vertex():
    return {
        "position": [1.0, -2.0, 3.0, 1.0],
        "normal": [0.0, 0.0, 1.0],
        "texcoords": [
            [0.1, 0.2, 0.3, 1.0],
            [0.4, 0.5, 0.6, 1.0],
        ],
        "color": [0.25, 0.5, 0.75, 1.0],
    }


def test_identity_transform_positions():
    dut = VertexTransform()
    t = SimpleTestbench(dut)

    mv = identity_mat(4)
    proj = identity_mat(4)
    vertex = make_vertex()

    mv_inv_t = inv(mv).T

    def as_amaranth_matrix(np_mat):
        return Cat(FixedPoint_mem.const(v) for v in np_mat.flatten())

    t.set_csrs(
        dut.csr_bus,
        [
            ((("position", "MV"),), as_amaranth_matrix(mv)),
            ((("position", "P"),), as_amaranth_matrix(proj)),
            ((("normal", "MV_inv_t"),), as_amaranth_matrix(mv_inv_t)),
            ((("enable",),), C(0b000)),
        ],
        "dut",
    )

    async def output_checker(ctx, results):
        assert len(results) == 1
        out = results[0]

        def vec_to_list(vec):
            return [c.as_float() for c in vec]

        assert vec_to_list(out.position_view) == pytest.approx(vertex["position"])
        assert vec_to_list(out.position_proj) == pytest.approx(vertex["position"])

        # Disabled normal/tex transforms should zero normals and leave texcoords as identity defaults (0,0,0,1)
        assert vec_to_list(out.normal_view) == pytest.approx([0.0, 0.0, 0.0])
        for tex_idx in range(2):
            assert vec_to_list(out.texcoords[tex_idx]) == pytest.approx(
                [0.0, 0.0, 0.0, 1.0]
            )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        input_stream=dut.is_vertex,
        input_data=[vertex],
        output_stream=dut.os_vertex,
        output_data_checker=output_checker,
        init_process=t.initialize_csrs,
        is_finished=dut.ready,
    )

    sim.run()
