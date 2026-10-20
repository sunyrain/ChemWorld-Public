# ChemWorld runtime architecture diagram

Generated with the built-in OpenAI ImageGen tool on 2026-10-20.
The diagram depicts the public runtime's configuration, interfaces and recorded history.
Final asset: `chemworld-runtime-architecture.png`.

## Generation prompt

```text
Use case: infographic-diagram.
Asset type: a finished architecture diagram image for the GitHub README of ChemWorld, an open-source simulation foundation. Create a crisp wide landscape image, approximately 2400 by 1200 pixels, aspect ratio 2:1. It must be readable when displayed at 1000 pixels wide.
Primary request: turn the following exact software architecture into a beautifully composed, scientifically sober visual. The diagram has five groups: Physical world, Task, ChemWorldRuntime, Experimental history, and a four-interface group. All wording below is exact.
Layout: two small cards stacked on the left; Physical world above Task. A larger central ChemWorldRuntime card vertically centered between them. A spacious Experimental history card on the right. Beneath the central card, a single horizontal Interfaces group with four evenly spaced pills: Python, Gym, CLI, Agent. The bottom group must not visually overlap any card or connector. Use generous whitespace and balanced alignment. Content occupies the full useful width with safe outer margins.
Connections, exactly: Physical world has one right-pointing arrow into ChemWorldRuntime. Task has a separate right-pointing arrow into ChemWorldRuntime. ChemWorldRuntime has one right-pointing arrow into Experimental history. The Interfaces group has one vertical double-headed arrow connecting to ChemWorldRuntime. There are no other arrows. Paths are clean, orthogonal where needed, with obvious arrowheads, no crossings and no floating endpoints. Do not imply four independent runtime engines.
Text, exact, grouped as follows:
Left upper card title: "Physical world"
Left upper card subtitle: "Materials · Equipment · Models"
Left lower card title: "Task"
Left lower card subtitle: "Goals · Permissions · Budgets"
Central card title: "ChemWorldRuntime"
Central card subtitle: "Execute · Measure · Record"
Right card title: "Experimental history"
Right card subtitles on two lines: "State · Observations" and "Resources · Replay"
Lower group label: "Interfaces"
Lower group pills: "Python", "Gym", "CLI", "Agent".
Style: premium editorial technical infographic in the same restrained identity as a modular scientific laboratory: warm ivory background #FAF7F1, deep ink/navy typography #102435, muted teal #247E82, and sparse copper #B9784C. Mostly flat technical drawing with subtle paper-like depth and extremely soft shadows. Large modern sans-serif typography with precise spelling. Central card has a pale teal fill and a stronger teal border; other cards have warm white surfaces and delicate borders. Arrow lines and arrowheads are dark teal and clearly visible. Small elegant line icons may accompany the headings: a vessel for Physical world, a target for Task, an interconnected process motif for ChemWorldRuntime, and stacked records for Experimental history. Icons must be subordinate to text.
Constraints: absolutely no extra headings, slogans, text, equations, code, numbers, watermarks, logos, graphs, flow stages or decorative arrows. Do not draw a screenshot, browser window or full web page. Do not use a dark background, neon colors, gradients or a 3D laboratory montage. Every label must be large enough to read in a README. Preserve all five groups and exactly the four specified connections. Render the finished diagram directly as one image.
```
