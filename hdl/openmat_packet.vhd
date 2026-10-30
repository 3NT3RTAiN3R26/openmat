-- openmat_packet.vhd
-- Byte-stream parser for one openmat HIL frame.
-- A register write updates scratch; a register read queues the scratch value.
-- Payload bytes of a sample write are pushed into the byte FIFO.
--
-- Revision History:
-- 2026-10-30  Tobias Zoske  Initial version.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.openmat_hil_pkg.all;

entity openmat_packet is
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    rx_valid  : in  std_logic;
    rx_data   : in  std_logic_vector(7 downto 0);
    tx_valid  : out std_logic;
    tx_data   : out std_logic_vector(7 downto 0);
    fifo_wr   : out std_logic;
    fifo_data : out std_logic_vector(7 downto 0);
    scratch   : out std_logic_vector(31 downto 0)
  );
end entity openmat_packet;

architecture rtl of openmat_packet is
  type t_state is (
    st_magic0, st_magic1, st_magic2, st_magic3,
    st_version, st_opcode, st_addr, st_len,
    st_payload, st_crc, st_reply
  );

  signal state     : t_state := st_magic0;
  signal opcode    : std_logic_vector(7 downto 0) := (others => '0');
  signal addr      : std_logic_vector(31 downto 0) := (others => '0');
  signal length    : std_logic_vector(31 downto 0) := (others => '0');
  signal crc       : std_logic_vector(31 downto 0) := (others => '1');
  signal got_crc   : std_logic_vector(31 downto 0) := (others => '0');
  signal count     : integer range 0 to 7 := 0;
  signal payload_n : integer range 0 to 255 := 0;
  signal scratch_r : std_logic_vector(31 downto 0) := (others => '0');
  signal reply_sel : integer range 0 to 3 := 0;
  signal crc_ok    : std_logic := '0';
begin
  scratch <= scratch_r;

  proc_parse : process (clk)
    variable v_crc : std_logic_vector(31 downto 0);
  begin
    if rising_edge(clk) then
      tx_valid  <= '0';
      tx_data   <= (others => '0');
      fifo_wr   <= '0';
      fifo_data <= (others => '0');

      if rst = '1' then
        state     <= st_magic0;
        opcode    <= (others => '0');
        addr      <= (others => '0');
        length    <= (others => '0');
        crc       <= (others => '1');
        got_crc   <= (others => '0');
        count     <= 0;
        payload_n <= 0;
        scratch_r <= (others => '0');
        reply_sel <= 0;
        crc_ok    <= '0';
      elsif rx_valid = '1' and state /= st_reply then
        v_crc := crc32_byte(crc, rx_data);
        case state is
          when st_magic0 =>
            crc <= (others => '1');
            if rx_data = C_MAGIC(31 downto 24) then
              state <= st_magic1;
              crc   <= crc32_byte(x"FFFFFFFF", rx_data);
            end if;
          when st_magic1 =>
            crc <= v_crc;
            if rx_data = C_MAGIC(23 downto 16) then
              state <= st_magic2;
            else
              state <= st_magic0;
            end if;
          when st_magic2 =>
            crc <= v_crc;
            if rx_data = C_MAGIC(15 downto 8) then
              state <= st_magic3;
            else
              state <= st_magic0;
            end if;
          when st_magic3 =>
            crc <= v_crc;
            if rx_data = C_MAGIC(7 downto 0) then
              state <= st_version;
            else
              state <= st_magic0;
            end if;
          when st_version =>
            crc <= v_crc;
            if rx_data = C_VERSION then
              state <= st_opcode;
            else
              state <= st_magic0;
            end if;
          when st_opcode =>
            crc    <= v_crc;
            opcode <= rx_data;
            count  <= 0;
            state  <= st_addr;
          when st_addr =>
            crc  <= v_crc;
            addr <= addr(23 downto 0) & rx_data;
            if count = 3 then
              count <= 0;
              state <= st_len;
            else
              count <= count + 1;
            end if;
          when st_len =>
            crc    <= v_crc;
            length <= length(23 downto 0) & rx_data;
            if count = 3 then
              count <= 0;
              if unsigned(length(23 downto 0) & rx_data) = 0 then
                state <= st_crc;
              else
                payload_n <= to_integer(unsigned((length(23 downto 0) & rx_data)(7 downto 0)));
                state <= st_payload;
              end if;
            else
              count <= count + 1;
            end if;
          when st_payload =>
            crc       <= v_crc;
            fifo_wr   <= '1';
            fifo_data <= rx_data;
            if opcode = C_OP_REGWRITE and count < 4 then
              scratch_r <= scratch_r(23 downto 0) & rx_data;
            end if;
            if count + 1 = payload_n or count = 7 then
              count <= 0;
              state <= st_crc;
            else
              count <= count + 1;
            end if;
          when st_crc =>
            got_crc <= got_crc(23 downto 0) & rx_data;
            if count = 3 then
              if (got_crc(23 downto 0) & rx_data) = (not crc) then
                crc_ok <= '1';
              else
                crc_ok <= '0';
              end if;
              reply_sel <= 0;
              state <= st_reply;
              count <= 0;
            else
              crc   <= v_crc;
              count <= count + 1;
            end if;
          when st_reply =>
            null;
        end case;
      elsif state = st_reply then
        tx_valid <= '1';
        case reply_sel is
          when 0 =>
            tx_data <= scratch_r(31 downto 24);
          when 1 =>
            tx_data <= scratch_r(23 downto 16);
          when 2 =>
            tx_data <= scratch_r(15 downto 8);
          when others =>
            tx_data <= scratch_r(7 downto 0);
            state   <= st_magic0;
            crc     <= (others => '1');
        end case;
        if reply_sel < 3 then
          reply_sel <= reply_sel + 1;
        else
          reply_sel <= 0;
        end if;
      end if;
    end if;
  end process proc_parse;
end architecture rtl;
