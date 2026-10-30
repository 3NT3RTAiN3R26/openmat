-- openmat_axi_regs.vhd
-- AXI4-Lite scratchpad used by the openmat HIL agent and JTAG-to-AXI.
-- Addresses: 0x0 scratch, 0x4 length, 0x8 status, 0xC control.
--
-- Revision History:
-- 2026-10-30  Tobias Zoske  Initial version.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity openmat_axi_regs is
  port (
    aclk          : in  std_logic;
    aresetn       : in  std_logic;
    s_axi_awaddr  : in  std_logic_vector(3 downto 0);
    s_axi_awvalid : in  std_logic;
    s_axi_awready : out std_logic;
    s_axi_wdata   : in  std_logic_vector(31 downto 0);
    s_axi_wstrb   : in  std_logic_vector(3 downto 0);
    s_axi_wvalid  : in  std_logic;
    s_axi_wready  : out std_logic;
    s_axi_bresp   : out std_logic_vector(1 downto 0);
    s_axi_bvalid  : out std_logic;
    s_axi_bready  : in  std_logic;
    s_axi_araddr  : in  std_logic_vector(3 downto 0);
    s_axi_arvalid : in  std_logic;
    s_axi_arready : out std_logic;
    s_axi_rdata   : out std_logic_vector(31 downto 0);
    s_axi_rresp   : out std_logic_vector(1 downto 0);
    s_axi_rvalid  : out std_logic;
    s_axi_rready  : in  std_logic;
    reg_scratch   : out std_logic_vector(31 downto 0);
    reg_length    : out std_logic_vector(31 downto 0);
    reg_status    : in  std_logic_vector(31 downto 0);
    reg_control   : out std_logic_vector(31 downto 0)
  );
end entity openmat_axi_regs;

architecture rtl of openmat_axi_regs is
  constant C_OKAY : std_logic_vector(1 downto 0) := "00";

  signal scratch : std_logic_vector(31 downto 0) := (others => '0');
  signal length  : std_logic_vector(31 downto 0) := (others => '0');
  signal control : std_logic_vector(31 downto 0) := (others => '0');

  signal awready : std_logic := '0';
  signal wready  : std_logic := '0';
  signal bvalid  : std_logic := '0';
  signal arready : std_logic := '0';
  signal rvalid  : std_logic := '0';
  signal rdata   : std_logic_vector(31 downto 0) := (others => '0');
  signal write_addr : std_logic_vector(3 downto 0) := (others => '0');
begin
  s_axi_awready <= awready;
  s_axi_wready  <= wready;
  s_axi_bresp   <= C_OKAY;
  s_axi_bvalid  <= bvalid;
  s_axi_arready <= arready;
  s_axi_rdata   <= rdata;
  s_axi_rresp   <= C_OKAY;
  s_axi_rvalid  <= rvalid;
  reg_scratch   <= scratch;
  reg_length    <= length;
  reg_control   <= control;

  proc_write : process (aclk)
    variable v_data : std_logic_vector(31 downto 0);
  begin
    if rising_edge(aclk) then
      if aresetn = '0' then
        awready    <= '0';
        wready     <= '0';
        bvalid     <= '0';
        write_addr <= (others => '0');
        scratch    <= (others => '0');
        length     <= (others => '0');
        control    <= (others => '0');
      else
        if awready = '0' and s_axi_awvalid = '1' then
          awready    <= '1';
          write_addr <= s_axi_awaddr;
        else
          awready <= '0';
        end if;

        if wready = '0' and s_axi_wvalid = '1' then
          wready <= '1';
          v_data := s_axi_wdata;
          if write_addr = "0000" then
            scratch <= v_data;
          elsif write_addr = "0100" then
            length <= v_data;
          elsif write_addr = "1100" then
            control <= v_data;
          end if;
        else
          wready <= '0';
        end if;

        if awready = '1' and wready = '1' and bvalid = '0' then
          bvalid <= '1';
        elsif bvalid = '1' and s_axi_bready = '1' then
          bvalid <= '0';
        end if;
      end if;
    end if;
  end process proc_write;

  proc_read : process (aclk)
  begin
    if rising_edge(aclk) then
      if aresetn = '0' then
        arready <= '0';
        rvalid  <= '0';
        rdata   <= (others => '0');
      else
        if arready = '0' and s_axi_arvalid = '1' then
          arready <= '1';
          if s_axi_araddr = "0000" then
            rdata <= scratch;
          elsif s_axi_araddr = "0100" then
            rdata <= length;
          elsif s_axi_araddr = "1000" then
            rdata <= reg_status;
          elsif s_axi_araddr = "1100" then
            rdata <= control;
          else
            rdata <= (others => '0');
          end if;
        else
          arready <= '0';
        end if;

        if arready = '1' and rvalid = '0' then
          rvalid <= '1';
        elsif rvalid = '1' and s_axi_rready = '1' then
          rvalid <= '0';
        end if;
      end if;
    end if;
  end process proc_read;
end architecture rtl;
