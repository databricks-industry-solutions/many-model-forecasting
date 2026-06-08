# Many Model Forecasting — Documentation Site

This folder contains the [Docusaurus](https://docusaurus.io/) source for the MMF documentation
website, published to GitHub Pages at:

> https://databricks-industry-solutions.github.io/many-model-forecasting/

## Local development

```bash
cd website
npm install      # installs dependencies (generates package-lock.json)
npm start        # starts a local dev server with hot reload at http://localhost:3000
```

## Production build

```bash
npm run build    # outputs a static site to website/build
npm run serve    # serves the production build locally
```

## Structure

```
website/
├── docs/                  # Markdown documentation pages (sidebar content)
├── src/
│   ├── css/custom.css     # Theme / branding overrides
│   └── pages/index.js     # Custom landing page
├── static/img/            # Favicon, social card, images
├── docusaurus.config.js   # Site configuration (URL, navbar, footer, theme)
├── sidebars.js            # Sidebar layout
└── package.json
```

## Deployment

Deployment is automated via GitHub Actions in
[`.github/workflows/deploy-docs.yml`](../.github/workflows/deploy-docs.yml). On every push to
`main` that touches `website/**`, the workflow builds the site and publishes it to GitHub Pages.

### One-time repository setup

In the GitHub repository settings, set **Settings → Pages → Build and deployment → Source** to
**GitHub Actions**. After the first successful workflow run, the site will be live at the URL above.

## Editing content

Each page under `docs/` is a Markdown file with front matter (`id`, `title`, `sidebar_position`).
The sidebar order is controlled by `sidebars.js`. Update `docusaurus.config.js` to change the
navbar, footer, or theme.
