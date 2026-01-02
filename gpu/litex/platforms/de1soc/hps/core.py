#
# This file is part of LiteX.
#
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-2-Clause

"""Altera/Intel HPS (Hard Processor System) - Cyclone V SoC hardcore CPU."""

import os

from litex.gen import *
from litex.soc.cores.cpu import CPU
from litex.soc.interconnect import axi
from migen import *


def quartus_axi3_params(prefix, axi_interface, dir="master"):
    """Generates a dictionary of Verilog instance parameters for an AXI3 interface."""
    params = {}

    o = "o" if dir == "master" else "i"
    i = "i" if dir == "master" else "o"

    # Write address channel
    params[f"{o}_{prefix}_awid"] = axi_interface.aw.id
    params[f"{o}_{prefix}_awaddr"] = axi_interface.aw.addr
    params[f"{o}_{prefix}_awlen"] = axi_interface.aw.len
    params[f"{o}_{prefix}_awsize"] = axi_interface.aw.size
    params[f"{o}_{prefix}_awburst"] = axi_interface.aw.burst
    params[f"{o}_{prefix}_awlock"] = axi_interface.aw.lock
    params[f"{o}_{prefix}_awcache"] = axi_interface.aw.cache
    params[f"{o}_{prefix}_awprot"] = axi_interface.aw.prot
    params[f"{o}_{prefix}_awvalid"] = axi_interface.aw.valid
    params[f"{i}_{prefix}_awready"] = axi_interface.aw.ready

    # Write data channel
    params[f"{o}_{prefix}_wid"] = axi_interface.w.id
    params[f"{o}_{prefix}_wdata"] = axi_interface.w.data
    params[f"{o}_{prefix}_wstrb"] = axi_interface.w.strb
    params[f"{o}_{prefix}_wlast"] = axi_interface.w.last
    params[f"{o}_{prefix}_wvalid"] = axi_interface.w.valid
    params[f"{i}_{prefix}_wready"] = axi_interface.w.ready

    # Write response channel
    params[f"{i}_{prefix}_bid"] = axi_interface.b.id
    params[f"{i}_{prefix}_bresp"] = axi_interface.b.resp
    params[f"{i}_{prefix}_bvalid"] = axi_interface.b.valid
    params[f"{o}_{prefix}_bready"] = axi_interface.b.ready

    # Read address channel
    params[f"{o}_{prefix}_arid"] = axi_interface.ar.id
    params[f"{o}_{prefix}_araddr"] = axi_interface.ar.addr
    params[f"{o}_{prefix}_arlen"] = axi_interface.ar.len
    params[f"{o}_{prefix}_arsize"] = axi_interface.ar.size
    params[f"{o}_{prefix}_arburst"] = axi_interface.ar.burst
    params[f"{o}_{prefix}_arlock"] = axi_interface.ar.lock
    params[f"{o}_{prefix}_arcache"] = axi_interface.ar.cache
    params[f"{o}_{prefix}_arprot"] = axi_interface.ar.prot
    params[f"{o}_{prefix}_arvalid"] = axi_interface.ar.valid
    params[f"{i}_{prefix}_arready"] = axi_interface.ar.ready

    # Read data channel
    params[f"{i}_{prefix}_rid"] = axi_interface.r.id
    params[f"{i}_{prefix}_rdata"] = axi_interface.r.data
    params[f"{i}_{prefix}_rresp"] = axi_interface.r.resp
    params[f"{i}_{prefix}_rlast"] = axi_interface.r.last
    params[f"{i}_{prefix}_rvalid"] = axi_interface.r.valid
    params[f"{o}_{prefix}_rready"] = axi_interface.r.ready

    return params


class HPS(CPU):
    variants = ["standard"]
    category = "hardcore"
    family = "arm"
    name = "hps"
    human_name = "HPS (Cyclone V SoC)"
    data_width = 32
    endianness = "little"
    reset_address = 0x0000_0000  # HPS boots from internal bootrom
    gcc_triple = "arm-none-eabi"
    gcc_flags = "-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard"
    linker_output_format = "elf32-littlearm"
    nop = "nop"
    io_regions = {0x4000_0000: 0x4000_0000}  # CSR region
    csr_decode = True  # AXI address decoded in AXI2Wishbone
    integrated_rom_supported = False  # HPS has internal bootrom

    # Memory Mapping
    @property
    def mem_map(self):
        return {
            "sram": 0x0000_0000,  # HPS DDR (via internal controller)
            "csr": 0x4000_0000,  # FPGA peripherals via h2f_axi
            "rom": 0xFFFF_0000,  # HPS bootrom
        }

    def __init__(self, platform, variant, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.platform = platform
        self.reset = Signal()
        self.periph_buses = []  # Peripheral buses (connected to main SoC bus)
        self.memory_buses = []  # Memory buses (to the HPS's DDR controller)

        # HPS AXI interfaces
        self.axi_gp_masters = []  # General Purpose AXI Masters (h2f_axi, h2f_axi_l)
        self.axi_hp_slaves = []  # High Performance AXI Slaves (f2h_sdram0)

        # # #

        # HPS Clocking (use system clock for now)
        self.cd_hps = ClockDomain()
        self.comb += self.cd_hps.clk.eq(ClockSignal("sys"))

        # HPS Instance parameters (minimal setup)
        hps_rst_n = Signal()
        ddr3_pads = platform.request("hps_ddr3")

        # IRQ signals (two 32-bit as HPS expects)
        self.irq0 = Signal(32)
        self.irq1 = Signal(32)

        self.cpu_params = {
            # Reset
            "o_h2f_rst_reset_n": hps_rst_n,
            # Clocks (use sys for all AXI interfaces)
            "i_h2f_axi_clk_clk": ClockSignal("hps"),
            "i_f2h_sdram_clk_clk": ClockSignal("hps"),
            # Interrupts (FPGA to HPS)
            "i_irq0_irq": self.irq0,
            "i_irq1_irq": self.irq1,
            # DDR3 pads (memory interface from HPS)
            "o_memory_mem_a": ddr3_pads.a,
            "o_memory_mem_ba": ddr3_pads.ba,
            "o_memory_mem_ck": ddr3_pads.ck_p,
            "o_memory_mem_ck_n": ddr3_pads.ck_n,
            "o_memory_mem_cke": ddr3_pads.cke,
            "o_memory_mem_cs_n": ddr3_pads.cs_n,
            "o_memory_mem_ras_n": ddr3_pads.ras_n,
            "o_memory_mem_cas_n": ddr3_pads.cas_n,
            "o_memory_mem_we_n": ddr3_pads.we_n,
            "o_memory_mem_reset_n": ddr3_pads.reset_n,
            "o_memory_mem_odt": ddr3_pads.odt,
            "o_memory_mem_dm": ddr3_pads.dm,
            "io_memory_mem_dq": ddr3_pads.dq,
            "io_memory_mem_dqs": ddr3_pads.dqs_p,
            "io_memory_mem_dqs_n": ddr3_pads.dqs_n,
            "i_memory_oct_rzqin": ddr3_pads.rzq,
        }

        # Add primary GP master (h2f_axi - HPS to FPGA AXI bridge)
        # Address width is 30-bit in hardware (1GB address space)
        self.pbus = axi.AXIInterface(
            data_width=32,
            address_width=32,  # (We ignore upper 2 bits)
            id_width=12,
            version="axi3",
            clock_domain="hps",
        )
        self.cpu_params.update(
            quartus_axi3_params("h2f_axi_master", self.pbus, dir="master")
        )
        self.axi_gp_masters.append(self.pbus)
        self.periph_buses.append(self.pbus)

        # Add f2h_sdram slave (FPGA to HPS DDR - for GPU access to main memory)
        # 64-bit data width, 8-bit ID width as per HPS configuration
        self.f2h_sdram = axi.AXIInterface(
            data_width=64,
            address_width=32,
            id_width=8,
            version="axi3",
            clock_domain="hps",
        )
        self.axi_hp_slaves.append(self.f2h_sdram)
        self.memory_buses.append(self.f2h_sdram)
        self.cpu_params.update(
            quartus_axi3_params("f2h_sdram_slave", self.f2h_sdram, dir="slave")
        )

        # Set reset
        self.comb += self.cd_hps.rst.eq(~hps_rst_n)

    def do_finalize(self):
        """Generate the HPS instance after all configuration is done."""
        # The HPS Verilog will be generated by Qsys - no source files needed here
        # Create HPS instance matching the Qsys-generated interface
        self.platform.add_source_dir(os.path.dirname(os.path.abspath(__file__)))
        self.specials += Instance("HPS", **self.cpu_params)
