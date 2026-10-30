# openmat

**Hinweis:** MATLAB ist eine Marke von MathWorks. openmat ist nicht mit MathWorks verbunden.

openmat ist ein Open-Source-Interpreter für einen MATLAB-ähnlichen Sprachumfang, geschrieben in Rust. Er führt `.m`-Skripte aus: Matrizen, Plots, Signalverarbeitung, Regelungstechnik, Fixed-Point und eine Hardware-in-the-Loop-Schnittstelle für FPGAs.

Lizenz: [Apache License 2.0](LICENSE). Copyright 2026 Tobias Zoske.

Das ist kein vollständiger MATLAB-Ersatz. Simulink und die proprietären Toolboxen fehlen. Der unterstützte Umfang steht unten.

## Bauen und starten

`rust-toolchain.toml` wählt Rust 1.99. Mit rustup reicht `cargo build`. Ohne rustup brauchst du Rust 1.85 oder neuer:

```bash
cargo build --workspace
cargo test --workspace
cargo run -p openmat-cli -- examples/linalg.m
cargo run -p openmat-cli -- -e "eig([2 0; 0 3])"
cargo run -p openmat-gui
cargo run -p openmat-gui -- examples/plot_sine.m
cargo run -p openmat-agent -- --proto udp --bind 0.0.0.0:5000
```

Ohne Datei startet `openmat` eine REPL, wenn das Terminal interaktiv ist. Sonst liest es stdin. Ein Pfad hinter `openmat-gui` lädt die Datei in den Editor und zeigt ihren Ordner links.

## GUI

Die Desktop-Oberfläche hat links den Dateibrowser, in der Mitte Editor, Konsole und Plot, rechts den Workspace.

**Files** listet den aktuellen Ordner mit Name, Size, Date Modified und Type. Ein Klick auf die Spaltenüberschrift sortiert. **Up** oder ein Doppelklick auf `..` geht eine Ebene nach oben. Ein Doppelklick auf einen Ordner öffnet ihn, ein Doppelklick auf eine `.m`-Datei lädt sie in den Editor. **Open…** und Strg+O öffnen den Dateidialog. **Save** und Strg+S schreiben die Datei; ohne Pfad fragt ein Dialog nach dem Namen. Eine Datei kann auch auf das Fenster gezogen werden.

Der Editor zeigt Zeilennummern. Ein Klick auf eine Nummer setzt oder entfernt einen Breakpoint (roter Punkt). Strg+I rückt `if`, `for`, `while`, `switch` und `function` ein und setzt mehrere Befehle, die durch `;` getrennt in einer Zeile stehen, auf eigene Zeilen. Semikolons in `[]` bleiben stehen, damit `[1, 2; 3, 4]` eine Matrix bleibt. Unter dem Editor steht **Lint**: Syntaxfehler, mehrere Befehle in einer Zeile, Tabs, Leerzeichen am Zeilenende und `break` / `continue` außerhalb einer Schleife. Ein fehlendes `;` ist kein Fehler. Die Zeile wird ausgeführt und ihr Ergebnis ausgegeben.

**Run** führt das Skript aus und hält an Breakpoints an. **Step** startet an der ersten Anweisung und geht danach eine Anweisung weiter, auch in Schleifen und Funktionen. Die aktuelle Zeile ist markiert, der Workspace zeigt die Variablen bis dahin. **Stop** bricht ab und behält die schon gesetzten Variablen. Während einer Sitzung ist der Editor gesperrt.

**Workspace** zeigt nach dem Lauf Name, Value, Size und Class. Skalare und kurze Vektoren stehen in der Value-Spalte. Ein Doppelklick öffnet die Variable als Tabelle, auch für `fi`, Cells und Structs. Konstanten wie `pi` und `i` werden nicht gelistet.

Plots aus `plot`, `stem`, `step` und `bode` erscheinen unter der Konsole.

## Sprache

- Ausdrücke, Zuweisungen, `ans`, Ausgabeunterdrückung mit `;`
- Matrixliterale: `[1 -2]` ist zwei Elemente, `[1 - 2]` und `[1-2]` sind Subtraktion
- 1-basierte Indizes, `end`, `:`
- `if` / `elseif` / `else`, `for`, `while`, `switch` / `case` / `otherwise`
- `function`-Dateien und anonyme Funktionen `@(x) x+1`
- Kommentare `%` und `%{ %}`, Zeilenfortsetzung `...`
- Werte: reelle und komplexe Matrizen (spaltenweise), Zeichenketten, logische Werte, Cells, Structs, Function-Handles, `fi`, `tf` / `ss`

## Funktionen

Lineare Algebra und Basisfunktionen: `zeros`, `ones`, `eye`, `size`, `length`, `numel`, `ndims`, `isempty`, `diag`, `linspace`, `reshape`, `sum`, `mean`, `min`, `max`, `inv`, `det`, `eig`, `svd`, `\`, `/`, `rand`, `randn`, `rng`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `floor`, `ceil`, `round`, `sign`, `real`, `imag`, `conj`, `transpose`, `ctranspose`, `disp`, `fprintf`, `sprintf`, `clear`, `home`.

`clear` entfernt Variablen aus dem Workspace. `clear name` entfernt einen Namen, `clear` und `clear all` leeren den Workspace. `pi`, `i`, `j`, `inf`, `nan`, `true` und `false` bleiben erhalten. `home` löscht die bisherige Konsolenausgabe; Text nach `home` bleibt sichtbar. Beide Funktionen gehen auch ohne Klammern: `clear x` und `home`.

Plots: `plot`, `stem`, `figure`, `hold`, `xlabel`, `ylabel`, `title`, `subplot`. Die GUI zeichnet die Figure unter der Konsole. Die CLI behält die Daten und bricht nicht ab.

Signalverarbeitung: `fft`, `ifft`, `filter`, `conv`, `freqz`, `butter`, `fir1`. `butter` benutzt die bilineare Transformation mit Vorverzerrung. `wn = 1` ist die Nyquist-Frequenz. Aufruf: `[b, a] = butter(n, wn)`.

Regelungstechnik: `tf`, `ss`, `step`, `bode`, `pole`. `step` integriert die Regelungsnormalform mit RK4.

Fixed-Point: `fi(wert, signed, wortbreite, nachkommastellen)`, `fi2hex`, `fi_write(x, 'datei.txt')`, `double`. Rundung ist kaufmännisch, Überlauf sättigt. `fi_write` schreibt ein Hex-Wort pro Zeile, spaltenweise, für VHDL-Testbenches.

## HIL

```matlab
h = hil.open('loopback');
h = hil.open('udp', '192.168.2.99', 5000);
h = hil.open('tcp', '192.168.2.99', 5000);
h = hil.open('jtag', 'pynq_rfsoc4x2');
hil.write(h, [1 2 3]);
y = hil.read(h, 3);
hil.regwrite(h, 0, 42);
v = hil.regread(h, 0);
hil.close(h);
```

Rahmen: Magic `OMAT`, Version, Opcode, Adresse, Länge, Nutzdaten, CRC-32. Opcodes: Write=1, Read=2, RegWrite=3, RegRead=4, Ack=5, Err=6. Abtastwerte sind little-endian `f64`.

Board-Profile liegen in `boards/`. `boards/pynq_rfsoc4x2.toml` beschreibt das PYNQ RFSoC 4x2 (XCZU48DR). `boards/explorer.toml` ist nur das Format für das FPGA-Horizons-Explorer-Board (Frühjahr 2027). Ein neues Board ist eine weitere TOML-Datei, kein Codewechsel. `OPENMAT_BOARDS` zeigt auf ein anderes Verzeichnis.

### PYNQ RFSoC 4x2

1. `openmat-agent` für `aarch64-unknown-linux-gnu` bauen und auf das Board kopieren. Unter Windows dient Zig als Linker, weil `aarch64-unknown-linux-gnu` ein Linux-`cc` erwartet.
2. Auf dem Board: `./openmat-agent --proto udp --bind 0.0.0.0:5000`. Die Meldung `listening on 0.0.0.0:5000 (udp)` heißt, dass der Agent bereit ist.
3. `hil.open('udp', host, port)` nimmt Host und Port aus dem Aufruf, nicht aus `ethernet.host` im Board-Profil. `examples/hil.m` spricht `192.168.178.98` Port 5000 an; die Adresse im Skript ist die des Boards. `fprintf` gibt `v` und `y` aus.
4. Der mitgelieferte Agent hält Register und Samples im Speicher. Die VHDL-Quellen unter `hdl/` (`openmat_axi_regs`, `openmat_fifo`, `openmat_packet`) sind der PL-Gegenpart: AXI-Registerbank, FIFO und Frame-Parser. Die Adressen 0, 4, 8 und 12 sind Scratch, Länge, Status und Control.

JTAG nutzt Vivado im Batch-Modus und JTAG-to-AXI. `OPENMAT_VIVADO` zeigt auf die `vivado`-Datei. Ohne Vivado bricht `hil.open('jtag', ...)` mit einer klaren Meldung ab. `hil.open('jtag-mock', 'pynq_rfsoc4x2')` prüft das Profil ohne Hardware. Sample-Ströme laufen über Ethernet, Registerzugriffe über Ethernet oder JTAG.

Der Loopback-Transport und die Protokolltests laufen ohne Board: `cargo run -p openmat-cli -- examples/hil_loopback.m`. Gegen das Board: `cargo run -p openmat-cli -- examples/hil.m`, während der Agent auf UDP-Port 5000 lauscht.

## Beispiele

- `examples/linalg.m`
- `examples/plot_sine.m`
- `examples/dsp_demo.m`
- `examples/fixed_point.m`
- `examples/hil_loopback.m`
- `examples/hil.m`
