---
name: frontend-design
description: Design system and frontend conventions for this project. Use whenever building or editing HTML templates, styling, or client-side JavaScript, so the interface stays consistent, modern, and professional.
---

# Frontend design system

This project's UI is server-rendered with Jinja2, styled with Tailwind CSS (via CDN),
and enhanced with vanilla JavaScript. No React, no npm, no build step — everything loads
from CDNs and is served directly by FastAPI. Match this stack exactly; do not introduce
a framework or a build tool.

## Stack
- Templating: Jinja2
- CSS: Tailwind via CDN (utility classes only)
- Icons: Lucide (CDN)
- Client JS: vanilla ES6, no framework
- Charts (only if needed): Chart.js (CDN)

## Visual language
- Dark theme by default: near-black background (zinc-950 / #0a0a0a), zinc-800 borders,
  zinc-100 text, and a single accent color used sparingly.
- Clean, ChatGPT-like layout: centered column, generous spacing, rounded-xl cards, subtle
  borders rather than heavy shadows.
- Two font weights only (normal, medium). Sentence case everywhere. No emoji in the UI.
- Fully responsive and mobile-friendly using Tailwind responsive prefixes.

## Interaction patterns
- Stream responses token-by-token with a blinking cursor.
- Always show loading / thinking states (animated dots) — never a frozen UI.
- Show metadata (model used, latency, sources) as small muted badges.
- Forms: clear focus states; disable the submit control while a request is in flight.

## Rules
- Keep styling as Tailwind utilities in the template; avoid separate stylesheets unless
  genuinely necessary.
- Keep JS minimal and readable — attach event listeners in a single script block, no
  inline `onclick` soup.
- Accessibility: label all inputs, ensure sufficient contrast, keep controls keyboard-usable.
- Reuse the same header, card, and button patterns across every page for consistency.

When building any new page, start from these conventions rather than a generic template.