from amaranth import *
from amaranth.lib import wiring
from amaranth.sim import Simulator
from amaranth_soc import csr
from amaranth_soc.csr.wishbone import WishboneCSRBridge
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Arbiter, Decoder
from amaranth_soc.wishbone.sram import WishboneSRAM

from gpu.input_assembly.cores import IndexGenerator
from gpu.utils.types import IndexKind

from ..utils.memory import DebugAccess, get_memory_resource


def test_index_generator():
    rst = Signal(1, reset_less=True)

    m = Module()

    m.submodules.dut = dut = IndexGenerator()
    m.submodules.csr_decoder = csr_decoder = csr.Decoder(addr_width=16, data_width=8)
    csr_decoder.add(dut.csr_bus, name="index_gen")

    m.submodules.csr_bridge = csr_bridge = WishboneCSRBridge(
        csr_decoder.bus, data_width=32
    )

    m.submodules.mem = mem = WishboneSRAM(
        size=1024, data_width=32, granularity=8, writable=True
    )
    m.submodules.decoder = decoder = Decoder(
        addr_width=32, data_width=32, granularity=8
    )
    m.submodules.arbiter = arbiter = Arbiter(
        addr_width=32, data_width=32, granularity=8
    )
    m.submodules.dbg_access = dbg_access = DebugAccess(
        addr_width=32, data_width=32, granularity=8
    )

    decoder.add(mem.wb_bus, addr=0x80000000)
    decoder.add(csr_bridge.wb_bus, addr=0x00000000)

    arbiter.add(dut.bus)
    arbiter.add(dbg_access.wb_bus)

    wiring.connect(m, arbiter.bus, decoder.bus)
    arbiter.bus.memory_map = decoder.bus.memory_map

    m = ResetInserter(rst)(m)
    mmap: MemoryMap = decoder.bus.memory_map

    async def test_with_data(
        ctx,
        addr: int,
        count: int,
        kind: IndexKind,
        memory_data: list[int],
        expected: list[int],
    ):
        # Reset DUT
        ctx.set(rst, 1)
        await ctx.tick()
        ctx.set(rst, 0)
        await ctx.tick()

        # setup memory
        print()
        print(f"Loading memory at {addr:#010x} with data: {memory_data}")
        await dbg_access.write(ctx, addr, memory_data)

        # Configure DUT
        print()
        print("Configuring DUT...")
        address_addr = get_memory_resource(mmap, "index_gen.address").start
        count_addr = get_memory_resource(mmap, "index_gen.count").start
        kind_addr = get_memory_resource(mmap, "index_gen.kind").start
        start_addr = get_memory_resource(mmap, "index_gen.start").start

        await ctx.tick().until(dut.ready)

        await dbg_access.write(ctx, address_addr, [addr])
        await dbg_access.write(ctx, count_addr, [count])
        await dbg_access.write(ctx, kind_addr, [kind])
        await dbg_access.write(ctx, start_addr, [1])

        await ctx.tick()

        # pull indices until ready
        indices = []
        while not ctx.get(dut.ready) or ctx.get(dut.os_index.valid):
            if ctx.get(dut.os_index.valid):
                indices.append(ctx.get(dut.os_index.payload))
                ctx.set(dut.os_index.ready, 1)
            else:
                ctx.set(dut.os_index.ready, 0)
            await ctx.tick().until(dut.ready | dut.os_index.valid)

        print(f"Generated indices: {indices}, expected: {expected}")
        # assert indices == expected, f"Expected indices {expected}, got {indices}"

    async def tb(ctx):
        await test_with_data(
            ctx,
            addr=0x80000000,
            count=10,
            kind=IndexKind.NOT_INDEXED,
            memory_data=[],
            expected=list(range(10)),
        )

        await test_with_data(
            ctx,
            addr=0x80000000,
            count=5,
            kind=IndexKind.U32,
            memory_data=[0, 2, 4, 1, 5],
            expected=[0, 2, 4, 1, 5],
        )

        await test_with_data(
            ctx,
            addr=0x80000000,
            count=6,
            kind=IndexKind.U16,
            memory_data=[0x00030002, 0x00050004, 0x00000001],
            expected=[2, 3, 4, 5, 1, 0],
        )

        await test_with_data(
            ctx,
            addr=0x80000000,
            count=8,
            kind=IndexKind.U8,
            memory_data=[0x04030201, 0x08070605],
            expected=[1, 2, 3, 4, 5, 6, 7, 8],
        )

        await test_with_data(
            ctx,
            addr=0x80000002,
            count=8,
            kind=IndexKind.U8,
            memory_data=[0x02010000, 0x06050403, 0x00000807],
            expected=[1, 2, 3, 4, 5, 6, 7, 8],
        )

    sim = Simulator(m)
    sim.add_clock(1e-9)
    sim.add_testbench(tb)
    with sim.write_vcd(
        "test_index_generator.vcd", "test_index_generator.gtkw", traces=dut
    ):
        sim.run()
