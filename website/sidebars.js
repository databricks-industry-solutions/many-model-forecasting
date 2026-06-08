// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: 'Model Families',
      collapsed: false,
      items: ['local-models', 'global-models', 'foundation-models'],
    },
    'mmf-agent',
    'examples',
    'project-support',
  ],
};

export default sidebars;
