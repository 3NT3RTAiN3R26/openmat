-- openmat_hil_pkg.vhd
-- Constants and CRC for the openmat hardware-in-the-loop frame.
--
-- Frame layout, little-endian multi-byte fields:
--   magic(4) = x"4F4D4154" ("OMAT"), version(1), opcode(1),
--   addr(4), length(4), payload(length), crc32(4).
--
-- Revision History:
-- 2026-10-30  Tobias Zoske  Initial version.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package openmat_hil_pkg is
  constant C_MAGIC  : std_logic_vector(31 downto 0) := x"4F4D4154";
  constant C_VERSION : std_logic_vector(7 downto 0) := x"01";

  constant C_OP_WRITE    : std_logic_vector(7 downto 0) := x"01";
  constant C_OP_READ     : std_logic_vector(7 downto 0) := x"02";
  constant C_OP_REGWRITE : std_logic_vector(7 downto 0) := x"03";
  constant C_OP_REGREAD  : std_logic_vector(7 downto 0) := x"04";
  constant C_OP_ACK      : std_logic_vector(7 downto 0) := x"05";
  constant C_OP_ERR      : std_logic_vector(7 downto 0) := x"06";

  function crc32_byte(
    crc  : std_logic_vector(31 downto 0);
    data : std_logic_vector(7 downto 0)
  ) return std_logic_vector;
end package openmat_hil_pkg;

package body openmat_hil_pkg is
  function crc32_byte(
    crc  : std_logic_vector(31 downto 0);
    data : std_logic_vector(7 downto 0)
  ) return std_logic_vector is
    variable v_crc : std_logic_vector(31 downto 0);
    variable v_bit : std_logic;
  begin
    v_crc := crc xor (x"000000" & data);
    for i in 0 to 7 loop
      v_bit := v_crc(0);
      v_crc := '0' & v_crc(31 downto 1);
      if v_bit = '1' then
        v_crc := v_crc xor x"EDB88320";
      end if;
    end loop;
    return v_crc;
  end function crc32_byte;
end package body openmat_hil_pkg;
