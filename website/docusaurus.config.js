// @ts-check
// Docusaurus configuration for the Many Model Forecasting documentation site.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Many Model Forecasting',
  tagline:
    'Bootstrap large-scale forecasting solutions on Databricks with local, global, and foundation time series models.',
  favicon: 'img/favicon.png',

  // Production URL of the site (GitHub Pages for the databricks-industry-solutions org).
  url: 'https://databricks-industry-solutions.github.io',
  // The repo name becomes the base path for project pages.
  baseUrl: '/many-model-forecasting/',

  // GitHub Pages deployment config.
  organizationName: 'databricks-industry-solutions',
  projectName: 'many-model-forecasting',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/databricks-industry-solutions/many-model-forecasting/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      colorMode: {
        defaultMode: 'light',
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'Many Model Forecasting',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {to: '/docs/intro', label: 'Get Started', position: 'left'},
          {
            href: 'https://github.com/databricks-industry-solutions/many-model-forecasting/tree/main/examples',
            label: 'Examples',
            position: 'left',
          },
          {
            href: 'https://github.com/databricks-industry-solutions/many-model-forecasting',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Introduction', to: '/docs/intro'},
              {label: 'Getting Started', to: '/docs/getting-started'},
              {label: 'MMF Agent', to: '/docs/mmf-agent'},
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'Example Notebooks',
                href: 'https://github.com/databricks-industry-solutions/many-model-forecasting/tree/main/examples',
              },
              {
                label: 'Vector Lab (YouTube)',
                href: 'https://www.youtube.com/@VectorLab',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/databricks-industry-solutions/many-model-forecasting',
              },
              {
                label: 'Databricks',
                href: 'https://www.databricks.com/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Databricks Industry Solutions. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'sql', 'yaml'],
      },
    }),
};

export default config;
