---
name: chart-visualization
description: This skill should be used when the user wants to visualize data. It intelligently selects the most suitable chart type from 26 available options, extracts parameters based on detailed specifications, and generates a chart image using a JavaScript script. Supports both static images and interactive HTML charts.
dependency:
  nodejs: ">=18.0.0"
---

# Chart Visualization Skill

This skill provides a comprehensive workflow for transforming data into visual charts. It handles chart selection, parameter extraction, and image generation. Supports both **static images** and **interactive HTML charts**.

## Workflow

To visualize data, follow these steps:

### 1. Intelligent Chart Selection
Analyze the user's data features to determine the most appropriate chart type. Use the following guidelines (and consult `references/` for detailed specs):

- **Time Series**: Use `generate_line_chart` (trends) or `generate_area_chart` (accumulated trends). Use `generate_dual_axes_chart` for two different scales.
- **Comparisons**: Use `generate_bar_chart` (categorical) or `generate_column_chart`. Use `generate_histogram_chart` for frequency distributions.
- **Part-to-Whole**: Use `generate_pie_chart` or `generate_treemap_chart` (hierarchical).
- **Relationships & Flow**: Use `generate_scatter_chart` (correlation), `generate_sankey_chart` (flow), or `generate_venn_chart` (overlap).
- **Maps**: Use `generate_district_map` (regions), `generate_pin_map` (points), or `generate_path_map` (routes).
- **Hierarchies & Trees**: Use `generate_organization_chart` or `generate_mind_map`.
- **Specialized**:
    - `generate_radar_chart`: Multi-dimensional comparison.
    - `generate_funnel_chart`: Process stages.
    - `generate_liquid_chart`: Percentage/Progress.
    - `generate_word_cloud_chart`: Text frequency.
    - `generate_boxplot_chart` or `generate_violin_chart`: Statistical distribution.
    - `generate_network_graph`: Complex node-edge relationships.
    - `generate_fishbone_diagram`: Cause-effect analysis.
    - `generate_flow_diagram`: Process flow.
    - `generate_spreadsheet`: Tabular data or pivot tables for structured data display and cross-tabulation.

### 2. Parameter Extraction
Once a chart type is selected, read the corresponding file in the `references/` directory (e.g., `references/generate_line_chart.md`) to identify the required and optional fields.
Extract the data from the user's input and map it to the expected `args` format.

### 3. Chart Generation
Invoke the `scripts/generate.js` script with a JSON payload.

**Payload Format (Static Image):**
```json
{
  "tool": "generate_chart_type_name",
  "args": {
    "data": [...],
    "title": "...",
    "theme": "...",
    "style": { ... }
  }
}
```

**Payload Format (Interactive HTML):**
```json
{
  "tool": "generate_chart_type_name",
  "args": {
    "data": [...],
    "title": "...",
    "theme": "...",
    "interactive": true,
    "style": { ... }
  }
}
```

**Execution Command:**
```bash
node ./scripts/generate.js '<payload_json>'
```

### 4. Result Return
The script will output the URL of the generated chart image (or HTML file for interactive charts).
Return the following to the user:
- The image/HTML URL.
- The complete `args` (specification) used for generation.

---

# Enhanced Capabilities (v2.0)

## Interactive Charts (NEW!)

In addition to static images, now you can generate **interactive HTML charts** with ECharts:

### When to Use Interactive Charts
- User wants to explore data (hover, zoom, click)
- Dashboard presentations
- Web embedding
- Data storytelling with drill-down

### Interactive Chart Types

All 26 chart types support interactive mode. Key additions:

| Chart Type | Use Case | Interactive Features |
|------------|----------|---------------------|
| Line/Area | Trends | Zoom, pan, tooltip |
| Bar/Column | Comparisons | Sort, filter, tooltip |
| Pie/Treemap | Composition | Drill-down, legend toggle |
| Scatter | Correlation | Zoom, brush selection |
| Map | Geographic | Zoom, pan, region click |
| Sankey | Flow | Hover highlight, flow focus |
| Network | Relationships | Drag nodes, zoom |
| Radar | Multi-dim comparison | Tooltip, legend |

### How to Generate Interactive Charts

**Step 1: Add `interactive: true` to args:**

```json
{
  "tool": "generate_line_chart",
  "args": {
    "data": [
      {"month": "Jan", "value": 120},
      {"month": "Feb", "value": 150},
      {"month": "Mar", "value": 180}
    ],
    "title": "Monthly Sales Trend",
    "interactive": true
  }
}
```

**Step 2: Execute:**
```bash
node ./scripts/generate.js '<payload>'
```

**Step 3: Output:**
- Returns HTML file path: `/mnt/user-data/outputs/chart_xxxxx.html`
- Open in browser for full interactivity
- Can be embedded in websites

### Interactive Features Available

| Feature | Description |
|---------|-------------|
| Tooltip | Hover to see exact values |
| Legend | Click to toggle series |
| Zoom/Pan | Drag to zoom, scroll to pan |
| DataZoom | Slider for range selection |
| Toolbox | Save as image, data view, etc. |
| Drill-down | Click to see details |
| Animation | Smooth transitions |

---

## 3D Charts (NEW!)

Generate 3D visualizations for impressive presentations:

### Available 3D Charts
- `generate_3d_bar_chart` - 3D bar comparison
- `generate_3d_scatter_chart` - 3D scatter plot
- `generate_3d_surface_chart` - 3D surface (for terrain/heatmaps)
- `generate_globe_chart` - Interactive globe for geographic data

### Usage
```json
{
  "tool": "generate_3d_bar_chart",
  "args": {
    "data": [...],
    "title": "3D Sales Comparison",
    "theme": "dark"
  }
}
```

---

## Real-time Dashboard (NEW!)

Create a dashboard with multiple interactive charts:

### Use Case
- Multi-metric monitoring
- Cross-chart filtering
- Time-range selection

### How to Create
1. Generate multiple interactive charts
2. Combine into dashboard HTML
3. Add cross-filter functionality

Example payload:
```json
{
  "tool": "generate_dashboard",
  "args": {
    "charts": [
      {"type": "line", "data": [...], "title": "Trend"},
      {"type": "bar", "data": [...], "title": "Comparison"},
      {"type": "pie", "data": [...], "title": "Distribution"}
    ],
    "layout": "grid",
    "theme": "light"
  }
}
```

---

## Geo Visualization (NEW!)

Enhanced map capabilities:

### Features
- **Choropleth Map**: Color-coded by value
- **Heat Map**: Density visualization
- **Flow Map**: Movement between locations
- **Bubble Map**: Size-coded points
- **Route Map**: Path visualization

### Data Format
```json
{
  "data": [
    {"name": "Beijing", "value": 100},
    {"name": "Shanghai", "value": 80},
    {"name": "Guangzhou", "value": 60}
  ],
  "geoJson": "china"  // or custom GeoJSON URL
}
```

---

## Usage Guidelines

### Choosing Chart Type

| Data Type | Static Image | Interactive |
|-----------|-------------|-------------|
| Simple comparison | ✅ Bar, Column | ✅ Bar, Column |
| Time series | ✅ Line, Area | ✅ Line, Area (recommended) |
| Composition | ✅ Pie, Treemap | ✅ Pie (recommended) |
| Correlation | ✅ Scatter | ✅ Scatter (recommended) |
| Geographic | ✅ Map | ✅ Map (recommended) |
| Complex relationships | ⚠️ Network | ✅ Network (recommended) |
| Dashboard | ❌ | ✅ Multiple charts |

### Best Practices

1. **Interactive for exploration** - Use interactive when user will explore data
2. **Static for reports** - Use static images for PDF reports
3. **3D sparingly** - 3D can distort perception, use for impact only
4. **Dashboard for monitoring** - Combine related metrics
5. **Mobile consideration** - Interactive charts work on mobile too

---

## Output Handling

### Static Images
- Format: PNG
- Resolution: 1920x1080 (default)
- Location: `/mnt/user-data/outputs/`

### Interactive HTML
- Format: HTML with ECharts
- Features: Full interactivity
- Location: `/mnt/user-data/outputs/chart_*.html`
- Can be opened in browser or embedded

### Sharing
- Use `present_files` tool to share with user
- For interactive, share the HTML file path
- Can be hosted on internal server

---

## Reference Material

Detailed specifications for each chart type are located in the `references/` directory. Consult these files to ensure the `args` passed to the script match the expected schema.

## License

This `SKILL.md` is provided by [antviz/chart-visualization-skills](https://github.com/antviz/chart-visualization-skills).
Licensed under the [MIT License](https://github.com/antviz/chart-visualization-skills/blob/master/LICENSE).
