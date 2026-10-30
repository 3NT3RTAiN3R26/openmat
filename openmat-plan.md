# MATLAB-kompatibler Interpreter in Rust ("openmat")

Ein eigener, kostenloser Interpreter für MATLAB-`.m`-Dateien in reinem Rust (kein Python, kein Octave) mit CLI und nativer GUI, Signalverarbeitung, Regelungstechnik, Fixed-Point und einer HIL-Schnittstelle zum FPGA. Der Funktionsumfang wächst schrittweise. Ein vollständiger MATLAB-Ersatz ist es nicht.

## Ziel und Abgrenzung

- Echte `.m`-Skripte und -Funktionen ausführen, ohne Octave und ohne Python. Parser und Interpreter sind komplett in Rust geschrieben.
- Schwerpunkte: Matrixrechnung, Plots, Signalverarbeitung, Regelungstechnik, Fixed-Point, HIL mit dem FPGA.
- Nicht enthalten: Simulink, proprietäre Toolboxen und 100 % Kompatibilität. Die Liste unterstützter Funktionen bleibt eine Teilmenge.
- In Rust gibt es kein fertiges SciPy. Filterentwurf (`butter`, `fir1`), Regelungstechnik (`tf`, `ss`, `bode`) und Ähnliches werden selbst auf Basis kleinerer Crates geschrieben.

## Architektur

```mermaid
flowchart LR
    mFile[".m Datei / Konsole"] --> lexer[Lexer + Parser]
    lexer --> ast[AST]
    ast --> interp[Interpreter]
    interp --> lib["Builtins: nalgebra, rustfft"]
    interp --> plot[Plot Backend egui_plot]
    interp --> hil["HIL Modul"]
    hil --> eth["Ethernet UDP/TCP"] --> agent["openmat-agent (ARM Linux)"] --> fpga[FPGA Board]
    hil --> jtag["JTAG (hw_server / Vivado Tcl)"] --> fpga
    cli[CLI] --> interp
    gui["GUI (eframe/egui)"] --> interp
```

Cargo-Workspace mit getrennten Crates:

- `openmat-core`: Lexer, Parser (handgeschrieben, Pratt-Parser), AST, Werte-Typen, Interpreter.
- `openmat-lib`: Builtins für lineare Algebra, DSP, Regelungstechnik, Fixed-Point.
- `openmat-hil`: Transport-Trait mit Ethernet-, JTAG- und Loopback-Backend, Rahmenprotokoll und Board-Profile.
- `openmat-agent`: kleiner Rust-Dienst für das ARM-Linux des PYNQ-Boards (aarch64), Zugriff auf AXI-Register und DMA.
- `openmat-cli`: Binärprogramm `openmat script.m`, REPL über `rustyline`.
- `openmat-gui`: Desktop-GUI mit `eframe`/`egui` und `egui_plot`.

Gewählte Crates: `nalgebra` (Matrizen, `eig`, `svd`, `inv`, `\`), `num-complex`, `rustfft`, `clap`, `rustyline`, `thiserror`.

Datenmodell: `Value`-Enum mit 2D-Matrix (reell und komplex, spaltenmajor), String, Bool/Logical, Cell-Array, Struct, Function-Handle. Alles ist mindestens 2D, Indizes sind 1-basiert.

MATLAB-Eigenheiten, die der Parser beachten muss: Transpose `'` gegen String, Matrixliterale mit Leerzeichen als Trenner (`[1 -2]` gegen `[1 - 2]`), `end` im Index, Kommentare `%` und Blockkommentare, Zeilenfortsetzung `...`, Ausgabe-Unterdrückung durch `;`.

## Schritte

1. **Kern**: Parser und Interpreter für Ausdrücke, Variablen, Matrixliterale, Indizierung, Ranges, `if`/`for`/`while`/`switch`, `function`-Dateien, anonyme Funktionen, `disp`/`fprintf`. Builtins `zeros`, `ones`, `eye`, `size`, `inv`, `\`, `eig`, `svd` und weitere. CLI und REPL. Tests mit `cargo test`.
2. **Plots und GUI**: `plot`, `figure`, `hold`, `xlabel`, `title`, `subplot`, `stem`. GUI mit Editor, Konsole, Workspace-Ansicht und Plot-Fenster.
3. **Signalverarbeitung und Regelungstechnik**: `fft`/`ifft` (über `rustfft`), `filter`, `conv`, `freqz`, `butter`, `fir1`, `tf`, `ss`, `step`, `bode`, `pole`. Teile davon werden selbst implementiert.
4. **Fixed-Point und FPGA-Modelle**: `fi`-ähnlicher Datentyp mit Wortbreite, Bruchteil, Rundung und Sättigung, bit-genau. Damit entstehen Referenzmodelle und Testvektor-Dateien für Vivado-Testbenches.
5. **HIL mit dem FPGA (Ethernet und JTAG von Anfang an)**: MATLAB-sichtbare Builtins `hil.open`, `hil.write`, `hil.read`, `hil.close`, `hil.regwrite`, `hil.regread`. Der Transport ist über ein Rust-Trait austauschbar:
   - **Ethernet (UDP/TCP)**: Rahmenprotokoll mit Header, Länge, Nutzdaten und CRC. Auf dem PYNQ RFSoC 4x2 läuft auf dem ARM-Linux ein kleiner Rust-Agent (`openmat-agent`). Er nimmt Pakete entgegen und schreibt oder liest AXI-Register und DMA-Puffer im PL.
   - **JTAG**: Anbindung über Xilinx `hw_server` (TCP, Port 3121) beziehungsweise Vivado im Batch-Modus mit Tcl und JTAG-to-AXI.
   - **Loopback-Mock**: für Tests ohne Hardware.
   - VHDL-Gegenstücke (AXI-Register-Bank, FIFO, Paket-Handling).
   - Board-Profile als Konfigurationsdatei (`boards/pynq_rfsoc4x2.toml`, später `boards/explorer.toml`).
6. **Doku**: README mit Installation, Liste unterstützter Funktionen und Beispiel-`.m`-Skripten.

## Festgelegte Annahmen

- Reines Rust, keine Python-Abhängigkeit. Die GUI ist eine native Desktop-App (egui).
- HIL unterstützt Ethernet und JTAG. Erstes Board ist das PYNQ RFSoC 4x2, später das FPGA-Horizons-Explorer-Board.
- Das Explorer-Board kommt erst im Frühjahr 2027. Bis dahin nur das Profilformat, Inhalte folgen mit Datenblatt oder Constraints.
- Der Test am echten PYNQ-Board läuft beim Nutzer. Hier nur Loopback-Test, Protokolllogik und README-Anleitung.
- Lizenz: Apache License 2.0. Hinweis: MATLAB ist eine Marke von MathWorks. openmat ist nicht mit MathWorks verbunden.
