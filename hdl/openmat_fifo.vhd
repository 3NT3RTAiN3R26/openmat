-- openmat_fifo.vhd
-- Synchronous FIFO with a fixed depth. Read data is registered.
--
-- Revision History:
-- 2026-10-30  Tobias Zoske  Initial version.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity openmat_fifo is
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    wr_en : in  std_logic;
    din   : in  std_logic_vector(7 downto 0);
    rd_en : in  std_logic;
    dout  : out std_logic_vector(7 downto 0);
    full  : out std_logic;
    empty : out std_logic
  );
end entity openmat_fifo;

architecture rtl of openmat_fifo is
  constant C_DEPTH : integer := 16;

  type t_mem is array (0 to C_DEPTH - 1) of std_logic_vector(7 downto 0);

  signal mem     : t_mem := (others => (others => '0'));
  signal wr_ptr  : integer range 0 to C_DEPTH - 1 := 0;
  signal rd_ptr  : integer range 0 to C_DEPTH - 1 := 0;
  signal count   : integer range 0 to C_DEPTH := 0;
  signal wr_fire : std_logic;
  signal rd_fire : std_logic;
begin
  wr_fire <= '1' when wr_en = '1' and count < C_DEPTH else '0';
  rd_fire <= '1' when rd_en = '1' and count > 0 else '0';
  full    <= '1' when count = C_DEPTH else '0';
  empty   <= '1' when count = 0 else '0';

  proc_write : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        wr_ptr <= 0;
        mem    <= (others => (others => '0'));
      elsif wr_fire = '1' then
        mem(wr_ptr) <= din;
        if wr_ptr = C_DEPTH - 1 then
          wr_ptr <= 0;
        else
          wr_ptr <= wr_ptr + 1;
        end if;
      end if;
    end if;
  end process proc_write;

  proc_read : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        rd_ptr <= 0;
        dout   <= (others => '0');
      else
        if rd_fire = '1' then
          dout <= mem(rd_ptr);
          if rd_ptr = C_DEPTH - 1 then
            rd_ptr <= 0;
          else
            rd_ptr <= rd_ptr + 1;
          end if;
        end if;
      end if;
    end if;
  end process proc_read;

  proc_count : process (clk)
    variable v_count : integer range 0 to C_DEPTH;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        count <= 0;
      else
        v_count := count;
        if wr_fire = '1' and rd_fire = '0' then
          v_count := v_count + 1;
        elsif wr_fire = '0' and rd_fire = '1' then
          v_count := v_count - 1;
        end if;
        count <= v_count;
      end if;
    end if;
  end process proc_count;
end architecture rtl;
