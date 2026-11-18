"use strict";
const fs = require("fs");
const path = require("path");

const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  Header,
  Footer,
  AlignmentType,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,
  VerticalAlign,
  PageNumber,
  PageBreak,
  LevelFormat,
  ImageRun,
} = require("docx");

// ── Colour constants ──────────────────────────────────────────────────────────
const DARK_BLUE  = "1F3864";
const MED_BLUE   = "2E75B6";
const LIGHT_BLUE = "DEEAF1";
const WHITE      = "FFFFFF";
const GRAY       = "808080";

// ── DXA helpers (1440 DXA = 1 inch) ──────────────────────────────────────────
const inch  = (n) => Math.round(n * 1440);
const pt    = (n) => n * 2; // half-points (1pt = 2 half-points)

// A4: 11906 × 16838 DXA, 1-inch margins → content width = 9026 DXA
const CONTENT_W = 11906 - inch(2); // 9026

// ── Shared border object ──────────────────────────────────────────────────────
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const borders = {
  top:    cellBorder,
  bottom: cellBorder,
  left:   cellBorder,
  right:  cellBorder,
};

// ── Utility: cell padding ─────────────────────────────────────────────────────
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// ── Header cell (dark blue bg, white bold text) ───────────────────────────────
function hdrCell(text, widthDxa) {
  return new TableCell({
    borders,
    width: { size: widthDxa, type: WidthType.DXA },
    shading: { fill: DARK_BLUE, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text, bold: true, color: WHITE, font: "Calibri", size: pt(10) })],
      }),
    ],
  });
}

// ── Data cell ────────────────────────────────────────────────────────────────
function dataCell(text, widthDxa, isAlt, isBold = false, isItalic = false, color = "000000") {
  return new TableCell({
    borders,
    width: { size: widthDxa, type: WidthType.DXA },
    shading: { fill: isAlt ? LIGHT_BLUE : WHITE, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold: isBold, italics: isItalic, color, font: "Calibri", size: pt(10) })],
      }),
    ],
  });
}

// ── Simple two-column table factory ──────────────────────────────────────────
function twoColTable(headers, rows, colWidths) {
  const [w1, w2] = colWidths;
  const tableRows = [
    new TableRow({
      tableHeader: true,
      children: [hdrCell(headers[0], w1), hdrCell(headers[1], w2)],
    }),
    ...rows.map(([c1, c2], i) =>
      new TableRow({
        children: [dataCell(c1, w1, i % 2 === 0), dataCell(c2, w2, i % 2 === 0)],
      })
    ),
  ];
  return new Table({
    width: { size: w1 + w2, type: WidthType.DXA },
    columnWidths: [w1, w2],
    rows: tableRows,
  });
}

// ── Four-column results table ─────────────────────────────────────────────────
function fourColTable(headers, rows, colWidths) {
  const tableRows = [
    new TableRow({
      tableHeader: true,
      children: headers.map((h, i) => hdrCell(h, colWidths[i])),
    }),
    ...rows.map((cells, ri) =>
      new TableRow({
        children: cells.map((c, ci) => dataCell(c, colWidths[ci], ri % 2 === 0)),
      })
    ),
  ];
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows,
  });
}

// ── Heading helpers ───────────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, color: DARK_BLUE, font: "Calibri", size: pt(16) })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, color: MED_BLUE, font: "Calibri", size: pt(13) })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    children: [
      new TextRun({
        text,
        font: "Calibri",
        size: pt(11),
        italics: opts.italic || false,
        color: opts.color || "000000",
      }),
    ],
  });
}

function spacer(before = 80, after = 80) {
  return new Paragraph({ spacing: { before, after }, children: [] });
}

function figPlaceholder(text) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    children: [new TextRun({ text, italics: true, color: GRAY, font: "Calibri", size: pt(10) })],
  });
}

const PROJECT_DIR = "C:\\Users\\HP\\OneDrive - University of Cape Town\\ACML\\Project";

function figImage(filename, dispW, dispH, caption) {
  const imgPath = path.join(PROJECT_DIR, filename);
  const imgData = fs.readFileSync(imgPath);
  const ext = path.extname(filename).slice(1).toLowerCase(); // "png"
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [
        new ImageRun({
          type: ext,
          data: imgData,
          transformation: { width: dispW, height: dispH },
          altText: { title: caption, description: caption, name: caption },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 160 },
      children: [new TextRun({ text: caption, italics: true, color: GRAY, font: "Calibri", size: pt(9) })],
    }),
  ];
}

function bulletItem(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Calibri", size: pt(11) })],
  });
}

function numberedItem(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Calibri", size: pt(11) })],
  });
}

// ── Page break ────────────────────────────────────────────────────────────────
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ── Build document ────────────────────────────────────────────────────────────
const doc = new Document({
  // ── Numbering ──────────────────────────────────────────────────────────────
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
      {
        reference: "numbered",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
      {
        reference: "worked",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
      {
        reference: "notworked",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
      {
        reference: "limitations",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
      {
        reference: "future",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { before: 40, after: 40 } } },
          },
        ],
      },
    ],
  },

  // ── Styles ─────────────────────────────────────────────────────────────────
  styles: {
    default: {
      document: { run: { font: "Calibri", size: pt(11) } },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: pt(16), bold: true, font: "Calibri", color: DARK_BLUE },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: pt(13), bold: true, font: "Calibri", color: MED_BLUE },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 },
      },
    ],
  },

  // ── Sections ───────────────────────────────────────────────────────────────
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: inch(1), right: inch(1), bottom: inch(1), left: inch(1) },
        },
      },
      headers: {
        default: new Header({ children: [] }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: pt(10), color: "666666" }),
              ],
            }),
          ],
        }),
      },

      children: [
        // ══════════════════════════════════════════════════════════════════════
        // COVER PAGE
        // ══════════════════════════════════════════════════════════════════════
        spacer(inch(1.5)),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 240 },
          children: [
            new TextRun({
              text: "S&P 500 Stock Price Forecasting using Deep Learning",
              bold: true,
              font: "Calibri",
              size: pt(24),
              color: DARK_BLUE,
            }),
          ],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 160 },
          children: [
            new TextRun({
              text: "COMS 4030A / COMS7047A — Adaptive Computation and Machine Learning",
              font: "Calibri",
              size: pt(14),
              color: "333333",
            }),
          ],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 400 },
          children: [
            new TextRun({ text: "University of Cape Town", font: "Calibri", size: pt(13), color: "444444" }),
          ],
        }),

        // Decorative rule
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 400 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: MED_BLUE, space: 1 } },
          children: [],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 120 },
          children: [
            new TextRun({ text: "Team Members", bold: true, font: "Calibri", size: pt(12), color: DARK_BLUE }),
          ],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 60 },
          children: [
            new TextRun({ text: "Zeeshaan Ebrahim", font: "Calibri", size: pt(12) }),
            new TextRun({ text: "  (2543004)", font: "Calibri", size: pt(11), color: "666666" }),
          ],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 400 },
          children: [
            new TextRun({ text: "Tazeem Tayob", font: "Calibri", size: pt(12) }),
            new TextRun({ text: "  (2659956)", font: "Calibri", size: pt(11), color: "666666" }),
          ],
        }),

        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 0 },
          children: [
            new TextRun({ text: "May 2026", font: "Calibri", size: pt(12), color: "444444" }),
          ],
        }),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 1: Dataset Description & Objectives
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("1. Dataset Description & Objectives"),

        body(
          "Source: Kaggle — S&P 500 Top 10 Stocks OHLCV (2010–2026). " +
          "The dataset contains split-adjusted daily trading data for nine S&P 500 technology companies, " +
          "spanning over 15 years of market history."
        ),

        spacer(120, 40),

        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [
            new TextRun({ text: "Table 1 — Dataset Columns", bold: true, font: "Calibri", size: pt(11) }),
          ],
        }),

        twoColTable(
          ["Column", "Description"],
          [
            ["Date",      "Trading date (YYYY-MM-DD)"],
            ["Ticker",    "Stock symbol (9 unique tickers)"],
            ["Open",      "Opening price (split-adjusted)"],
            ["High",      "Daily high price"],
            ["Low",       "Daily low price"],
            ["Close",     "Closing price"],
            ["Adj_Close", "Adjusted closing price (primary target)"],
            ["Volume",    "Number of shares traded"],
          ],
          [Math.round(CONTENT_W * 0.3), Math.round(CONTENT_W * 0.7)]
        ),

        spacer(120, 60),

        body(
          "Tickers: AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA"
        ),

        body(
          "Dataset statistics: 35,765 rows | 9 stocks | Date range: 2010-01-04 to 2026-02-13 | Missing values: 0"
        ),

        spacer(160, 80),
        h2("Objectives"),

        numberedItem(
          "Build a single unified deep learning model to forecast next trading day adjusted closing price for any of the nine stocks simultaneously.",
          "numbered"
        ),
        numberedItem(
          "Compare four architectures: Stacked LSTM, Bidirectional LSTM with Attention, CNN-LSTM Hybrid, and custom Transformer Encoder.",
          "numbered"
        ),
        numberedItem(
          "Investigate how window size, feature richness, dropout, prediction horizon, and L2 regularisation affect forecast accuracy.",
          "numbered"
        ),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 2: Preprocessing & Data Splits
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("2. Preprocessing & Data Splits"),

        h2("Feature Engineering"),

        body(
          "17 additional technical indicators were computed per stock using only information available up to time t " +
          "(no lookahead bias). All rolling calculations were performed per-ticker to prevent cross-stock contamination. " +
          "The RSI and Bollinger Band indicators were implemented from scratch."
        ),

        spacer(120, 40),

        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [
            new TextRun({ text: "Table 2 — Engineered Features", bold: true, font: "Calibri", size: pt(11) }),
          ],
        }),

        twoColTable(
          ["Feature Group", "Features"],
          [
            ["Trend (SMA)",  "SMA_5, SMA_20, SMA_50"],
            ["Trend (EMA)",  "EMA_12, EMA_26, MACD"],
            ["Momentum",     "RSI_14"],
            ["Volatility",   "BB_upper, BB_lower, BB_bandwidth"],
            ["Volume",       "Volume_MA_10"],
            ["Returns",      "Daily_Return, Volatility_10"],
            ["Identity",     "Ticker one-hot encoding (9 dimensions)"],
          ],
          [Math.round(CONTENT_W * 0.35), Math.round(CONTENT_W * 0.65)]
        ),

        spacer(100, 60),

        body("Total: 28 features per timestep."),

        spacer(160, 80),
        h2("Normalisation"),

        body(
          "MinMaxScaler was fitted exclusively on the training slice of each stock to prevent data leakage. " +
          "The same fitted scaler was then applied to the validation and test slices. Scalers were stored per-ticker " +
          "for inverse-transformation of predictions back to actual dollar prices."
        ),

        spacer(160, 80),
        h2("Data Splits"),

        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [
            new TextRun({ text: "Table 3 — Data Splits", bold: true, font: "Calibri", size: pt(11) }),
          ],
        }),

        (() => {
          const w1 = Math.round(CONTENT_W * 0.12);
          const w2 = Math.round(CONTENT_W * 0.28);
          const w3 = Math.round(CONTENT_W * 0.15);
          const w4 = CONTENT_W - w1 - w2 - w3;
          const hdrs = ["Split", "Date Range", "Sequences", "Purpose"];
          const rows = [
            ["Train",      "2010-01-04 to 2022-12-31", "27,746", "Fit models and learn weights"],
            ["Validation", "2023-01-01 to 2024-06-30", "2,826",  "Hyperparameter tuning and early stopping"],
            ["Test",       "2024-07-01 to 2026-02-13", "3,132",  "Final evaluation only (never seen during training)"],
          ];
          const tableRows = [
            new TableRow({
              tableHeader: true,
              children: [hdrCell(hdrs[0], w1), hdrCell(hdrs[1], w2), hdrCell(hdrs[2], w3), hdrCell(hdrs[3], w4)],
            }),
            ...rows.map(([c1, c2, c3, c4], i) =>
              new TableRow({
                children: [
                  dataCell(c1, w1, i % 2 === 0),
                  dataCell(c2, w2, i % 2 === 0),
                  dataCell(c3, w3, i % 2 === 0),
                  dataCell(c4, w4, i % 2 === 0),
                ],
              })
            ),
          ];
          return new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [w1, w2, w3, w4],
            rows: tableRows,
          });
        })(),

        spacer(120, 60),

        h2("Sequence Creation"),

        body(
          "A sliding window of 60 trading days (~3 months) was used. For each sample, " +
          "X = features for days [t-60, t-1] and y = normalised Adj_Close at day t. " +
          "Sequences from all nine stocks were stacked into a single dataset."
        ),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 3: Model Selection & Implementation
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("3. Model Selection & Implementation"),

        body(
          "All models were implemented in PyTorch 2.2.2 and trained with the Adam optimiser (lr=1e–3), " +
          "MSE loss, gradient clipping (max norm=1.0), EarlyStopping (patience=10), and ReduceLROnPlateau " +
          "(factor=0.5, patience=5)."
        ),

        spacer(160, 80),
        h2("Model 1 — Stacked LSTM (Baseline)"),

        body(
          "A two-layer stacked LSTM with dropout regularisation serves as the baseline. " +
          "Architecture: Input(60, 28) → LSTM(128, return_seq=True) → Dropout(0.2) → " +
          "LSTM(64) → Dropout(0.2) → Linear(32) → ReLU → Linear(1)."
        ),
        body("Parameters: 132,673     |     Best validation loss: 0.070183"),

        spacer(160, 80),
        h2("Model 2 — Bidirectional LSTM with Bahdanau Attention"),

        body(
          "Bidirectional LSTMs capture both forward and backward temporal context within the 60-day window. " +
          "A custom Bahdanau-style attention mechanism computes a weighted sum of hidden states, allowing the " +
          "model to focus on the most informative timesteps. " +
          "Architecture: Input → BiLSTM(128) → Dropout(0.2) → BiLSTM(64) → Dropout(0.2) → " +
          "Attention → context vector → Linear(32) → ReLU → Linear(1)."
        ),
        body("Parameters: 330,946     |     Best validation loss: 0.115797"),

        spacer(160, 80),
        h2("Model 3 — CNN-LSTM Hybrid"),

        body(
          "Temporal causal convolutions act as local pattern extractors detecting short-term momentum shifts, " +
          "while the LSTM captures long-range sequential dependencies. Causal padding (left-padding only) " +
          "prevents any lookahead bias. " +
          "Architecture: Input → CausalConv1D(64, k=3) ×2 → MaxPool1D(2) → " +
          "CausalConv1D(32, k=3) → LSTM(64) → Dropout(0.2) → Linear(32) → ReLU → Linear(1)."
        ),
        body("Parameters: 51,169     |     Best validation loss: 0.012034"),

        spacer(160, 80),
        h2("Model 4 — Custom Transformer Encoder"),

        body(
          "Built entirely from scratch using PyTorch primitives following the Vaswani et al. (2017) encoder " +
          "design with multi-head self-attention, position-wise feed-forward networks, layer normalisation, and " +
          "residual connections. Learned positional embeddings replace sinusoidal ones. " +
          "Architecture: Input → Linear(64) + PositionalEmbedding → TransformerBlock ×2 " +
          "[MultiHeadAttention(4 heads) + FFN(128) + LayerNorm] → GlobalAvgPool → " +
          "Linear(64) → Dropout(0.1) → Linear(1)."
        ),
        body("Parameters: 76,865     |     Best validation loss: 0.122278"),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 4: Experiments & Fine-Tuning
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("4. Experiments & Fine-Tuning"),

        body(
          "Five controlled experiments were conducted using the LSTM as a baseline " +
          "(except Experiment 5 which used the Transformer)."
        ),

        // Experiment 1
        spacer(160, 80),
        h2("Experiment 1 — Window Size"),

        body(
          "The sequence length was varied across 20, 30, and 60 trading days " +
          "to assess the impact of lookback horizon on forecast accuracy."
        ),

        spacer(80, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 4 — Window Size vs. Validation RMSE", bold: true, font: "Calibri", size: pt(11) })],
        }),
        twoColTable(
          ["Window Size", "Val RMSE"],
          [
            ["20 days",         "0.1599"],
            ["30 days",         "0.1952"],
            ["60 days (best)",  "0.0749"],
          ],
          [Math.round(CONTENT_W * 0.5), Math.round(CONTENT_W * 0.5)]
        ),
        spacer(80, 40),
        body(
          "Result: A 60-day window significantly outperformed shorter windows, capturing quarterly price trends " +
          "and momentum cycles invisible at shorter horizons."
        ),

        // Experiment 2
        spacer(160, 80),
        h2("Experiment 2 — Feature Ablation"),

        body(
          "OHLCV-only (6 features) was compared against the full technical indicator set (28 features)."
        ),

        spacer(80, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 5 — Feature Set vs. Validation RMSE", bold: true, font: "Calibri", size: pt(11) })],
        }),
        twoColTable(
          ["Feature Set", "Val RMSE"],
          [
            ["OHLCV-only (6 features)",                  "Higher RMSE"],
            ["Full features with indicators (28 features)", "Lower RMSE (winner)"],
          ],
          [Math.round(CONTENT_W * 0.55), Math.round(CONTENT_W * 0.45)]
        ),
        spacer(80, 40),
        body(
          "Result: Technical indicators (MACD, RSI, Bollinger Bands) provided meaningful additional signal, " +
          "reducing RMSE compared to raw price and volume data alone."
        ),

        // Experiment 3
        spacer(160, 80),
        h2("Experiment 3 — Dropout Rate"),

        body("Three dropout rates were tested on the LSTM."),

        spacer(80, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 6 — Dropout Rate vs. Validation RMSE", bold: true, font: "Calibri", size: pt(11) })],
        }),
        twoColTable(
          ["Dropout Rate", "Val RMSE"],
          [
            ["0.1 (best)", "0.1325"],
            ["0.2",        "0.2322"],
            ["0.3",        "0.3171"],
          ],
          [Math.round(CONTENT_W * 0.5), Math.round(CONTENT_W * 0.5)]
        ),
        spacer(80, 40),
        body(
          "Result: A dropout rate of 0.1 achieved the lowest validation RMSE. " +
          "Higher dropout rates caused under-fitting."
        ),

        // Experiment 4
        spacer(160, 80),
        h2("Experiment 4 — Prediction Horizon"),

        body("1-day and 5-day ahead forecasting were compared on the LSTM."),

        spacer(80, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 7 — Prediction Horizon vs. Validation RMSE", bold: true, font: "Calibri", size: pt(11) })],
        }),
        twoColTable(
          ["Horizon", "Val RMSE"],
          [
            ["1 day ahead (winner)", "Lower RMSE"],
            ["5 days ahead",         "Higher RMSE (~doubled)"],
          ],
          [Math.round(CONTENT_W * 0.5), Math.round(CONTENT_W * 0.5)]
        ),
        spacer(80, 40),
        body(
          "Result: The 5-day horizon was significantly harder, with RMSE roughly doubling, " +
          "indicating rapid error accumulation over multiple timesteps."
        ),

        // Experiment 5
        spacer(160, 80),
        h2("Experiment 5 — L2 Regularisation (Transformer)"),

        body("Weight decay was applied to the Transformer optimizer across three settings."),

        spacer(80, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 8 — Weight Decay vs. Validation RMSE (Transformer)", bold: true, font: "Calibri", size: pt(11) })],
        }),
        twoColTable(
          ["Weight Decay", "Val RMSE"],
          [
            ["0 (none)",      "0.3389"],
            ["1e-4 (best)",   "0.3250"],
            ["1e-3",          "0.3484"],
          ],
          [Math.round(CONTENT_W * 0.5), Math.round(CONTENT_W * 0.5)]
        ),
        spacer(80, 40),
        body(
          "Result: Mild L2 regularisation (1e-4) slightly improved generalisation. " +
          "Strong regularisation (1e-3) caused under-fitting."
        ),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 5: Training Visualisations
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("5. Training Visualisations"),

        body(
          "This section presents graphs generated during model training and experiments. " +
          "Figures should be inserted at the placeholder locations below."
        ),

        spacer(120, 80),

        ...figImage("fig_01_price_history.png", 580, 370, "Figure 1: Normalised adjusted closing prices for all 9 stocks (2010–2026)"),
        ...figImage("fig_02_correlation.png",   500, 389, "Figure 2: Cross-stock return correlation heatmap"),
        ...figImage("fig_03_volume.png",         580, 230, "Figure 3: Daily trading volume distribution by stock"),
        ...figImage("fig_04_loss_curves.png",    580, 339, "Figure 4: Training and validation MSE loss curves for all four models"),
        ...figImage("fig_05_experiments.png",    580, 141, "Figure 5: Experiment results — window size, feature ablation, dropout rate"),
        ...figImage("fig_06_experiments2.png",   580, 207, "Figure 6: Experiment results — prediction horizon and L2 regularisation"),
        ...figImage("fig_07_attention.png",      580,  78, "Figure 7: BiLSTM attention weight heatmap over 60-day lookback window"),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 6: Test Results & Analysis
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("6. Test Results & Analysis"),

        body(
          "All predictions were inverse-transformed from normalised [0,1] space back to actual dollar prices " +
          "before computing metrics."
        ),

        spacer(120, 40),
        new Paragraph({
          spacing: { before: 40, after: 60 },
          children: [new TextRun({ text: "Table 9 — Test Set Performance Across All Models", bold: true, font: "Calibri", size: pt(11) })],
        }),

        (() => {
          const w1 = Math.round(CONTENT_W * 0.24);
          const w2 = Math.round(CONTENT_W * 0.19);
          const w3 = Math.round(CONTENT_W * 0.19);
          const w4 = Math.round(CONTENT_W * 0.19);
          const w5 = CONTENT_W - w1 - w2 - w3 - w4;
          const hdrs = ["Model", "RMSE ($)", "MAE ($)", "MAPE (%)", "Dir. Accuracy (%)"];
          const rows = [
            ["LSTM",          "79.953", "55.399", "19.857", "49.38"],
            ["BiLSTM+Attn",   "98.377", "69.123", "23.934", "50.40"],
            ["CNN-LSTM",      "41.827", "26.785", "10.278", "51.17"],
            ["Transformer",  "101.240", "71.212", "24.502", "50.46"],
          ];
          const tableRows = [
            new TableRow({
              tableHeader: true,
              children: [hdrCell(hdrs[0], w1), hdrCell(hdrs[1], w2), hdrCell(hdrs[2], w3), hdrCell(hdrs[3], w4), hdrCell(hdrs[4], w5)],
            }),
            ...rows.map(([c1, c2, c3, c4, c5], i) => {
              const isCNN = c1 === "CNN-LSTM";
              return new TableRow({
                children: [
                  dataCell(c1, w1, i % 2 === 0, isCNN),
                  dataCell(c2, w2, i % 2 === 0, isCNN),
                  dataCell(c3, w3, i % 2 === 0, isCNN),
                  dataCell(c4, w4, i % 2 === 0, isCNN),
                  dataCell(c5, w5, i % 2 === 0, isCNN),
                ],
              });
            }),
          ];
          return new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [w1, w2, w3, w4, w5],
            rows: tableRows,
          });
        })(),

        spacer(80, 40),
        body(
          "Best model: CNN-LSTM with RMSE $41.83, MAE $26.79, MAPE 10.28%, Directional Accuracy 51.17%."
        ),

        spacer(120, 60),
        ...figImage("fig_08_metrics_comparison.png", 580, 172, "Figure 8: Test set performance comparison — RMSE, MAE, MAPE across all models"),
        ...figImage("fig_09_predictions.png",        560, 372, "Figure 9: Actual vs. predicted Adj_Close for AAPL, NVDA, TSLA (test set)"),
        ...figImage("fig_10_residuals.png",          560, 357, "Figure 10: Residual error distribution per stock — CNN-LSTM (best model)"),

        spacer(120, 80),
        h2("Analysis"),

        body(
          "The CNN-LSTM hybrid achieved the best performance across all metrics by a significant margin. " +
          "Its RMSE of $41.83 was roughly half that of the LSTM ($79.95) and less than half that of the Transformer ($101.24). " +
          "This suggests that the 1D causal convolutional layers are highly effective at extracting local momentum " +
          "patterns from the 60-day window before passing extracted features to the LSTM for sequential modelling."
        ),

        body(
          "The Bidirectional LSTM with Attention underperformed relative to expectations. While attention mechanisms " +
          "are theoretically beneficial, the added model complexity (330K parameters, 2.5× more than the LSTM) appears " +
          "to have caused overfitting, reflected in a higher validation loss (0.1158 vs 0.0702 for the LSTM)."
        ),

        body(
          "The Transformer Encoder was the weakest performer ($101.24 RMSE). Without pre-training on large datasets, " +
          "Transformers tend to require more data to generalise — the approximately 27,000 training sequences may be " +
          "insufficient for a pure attention-based approach."
        ),

        body(
          "All four models achieved directional accuracy close to 50%, slightly better than random chance. " +
          "This is consistent with the Efficient Market Hypothesis, which suggests short-term price direction is " +
          "inherently difficult to predict from historical data alone."
        ),

        body(
          "Residual distributions for all stocks are approximately centred around zero, indicating no systematic bias. " +
          "However, residuals are wider for high-volatility stocks (NVDA, TSLA) compared to more stable stocks (MSFT, AAPL)."
        ),

        // ══════════════════════════════════════════════════════════════════════
        // SECTION 7: Conclusions
        // ══════════════════════════════════════════════════════════════════════
        pageBreak(),

        h1("7. Conclusions"),

        body(
          "Four deep learning architectures were trained on a unified dataset of nine S&P 500 technology stocks " +
          "to forecast next-day adjusted closing prices. The CNN-LSTM hybrid was the clear best performer, achieving " +
          "a test RMSE of $41.83 and MAPE of 10.28%."
        ),

        spacer(160, 80),
        h2("What Worked"),

        bulletItem(
          "Engineering technical indicators (MACD, RSI, Bollinger Bands) reduced RMSE compared to raw OHLCV data alone",
          "worked"
        ),
        bulletItem(
          "A 60-day lookback window significantly outperformed 20- and 30-day windows, capturing quarterly trend cycles",
          "worked"
        ),
        bulletItem(
          "Causal convolutional feature extraction (CNN-LSTM) was more effective than pure recurrent or attention-based approaches",
          "worked"
        ),
        bulletItem(
          "ReduceLROnPlateau scheduling and early stopping prevented overfitting and reduced training time",
          "worked"
        ),

        spacer(160, 80),
        h2("What Did Not Work"),

        bulletItem(
          "The 5-day prediction horizon was significantly harder, with RMSE roughly doubling",
          "notworked"
        ),
        bulletItem(
          "High dropout rates (0.3) degraded performance, suggesting under-fitting beyond this threshold",
          "notworked"
        ),
        bulletItem(
          "Strong L2 regularisation (weight_decay=1e-3) caused under-fitting in the Transformer",
          "notworked"
        ),
        bulletItem(
          "The Bidirectional LSTM with Attention overfitted despite dropout regularisation",
          "notworked"
        ),

        spacer(160, 80),
        h2("Limitations"),

        bulletItem(
          "No fundamental data (earnings, macroeconomic indicators) used; models rely solely on technical price history",
          "limitations"
        ),
        bulletItem(
          "Financial time-series are non-stationary; concept drift events pose challenges",
          "limitations"
        ),
        bulletItem(
          "Directional accuracy near 50% limits practical trading utility",
          "limitations"
        ),

        spacer(160, 80),
        h2("Future Work"),

        bulletItem(
          "Cross-stock attention: allow each stock to attend to signals from the other eight",
          "future"
        ),
        bulletItem(
          "Multi-step recursive forecasting with uncertainty quantification (Monte Carlo Dropout)",
          "future"
        ),
        bulletItem(
          "Ensemble of all four architectures for lower-variance combined predictions",
          "future"
        ),
        bulletItem(
          "Incorporate fundamental and macroeconomic features",
          "future"
        ),

      ], // end children
    },  // end section
  ],    // end sections
});

// ── Write to disk ─────────────────────────────────────────────────────────────
const OUTPUT = path.join(
  "C:\\Users\\HP\\OneDrive - University of Cape Town\\ACML\\Project",
  "ACML_Report_v2.docx"
);

Packer.toBuffer(doc)
  .then((buffer) => {
    fs.writeFileSync(OUTPUT, buffer);
    console.log("Document written to:", OUTPUT);
    console.log("File size:", (buffer.length / 1024).toFixed(1), "KB");
  })
  .catch((err) => {
    console.error("Error generating document:", err);
    process.exit(1);
  });
