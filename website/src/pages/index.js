import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

const FEATURES = [
  {
    title: 'Local Models',
    description:
      'Fit individual time series with statistical models from statsforecast and sktime. Great interpretability and low data requirements.',
    link: '/docs/local-models',
  },
  {
    title: 'Global Models',
    description:
      'Train one model across many series with mlforecast (LightGBM) and neuralforecast deep learning models, with optional hyperparameter tuning.',
    link: '/docs/global-models',
  },
  {
    title: 'Foundation Models',
    description:
      'Run pretrained transformer models like Chronos-2 and TimesFM 2.5 with zero training, on GPU or serverless GPU compute.',
    link: '/docs/foundation-models',
  },
  {
    title: 'Configuration over Code',
    description:
      'Forecast hundreds or thousands of series in parallel on Spark with a single run_forecast call — minimal coding, fully extensible.',
    link: '/docs/getting-started',
  },
  {
    title: 'MLflow Integration',
    description:
      'Backtesting, evaluation, model logging, and registration to Unity Catalog are built in and tracked end to end in MLflow.',
    link: '/docs/getting-started',
  },
  {
    title: 'MMF Agent',
    description:
      'Drive an end-to-end forecasting project through natural language with skills for Claude Code, Cursor, and GitHub Copilot.',
    link: '/docs/mmf-agent',
  },
];

function Feature({title, description, link}) {
  return (
    <div className={clsx('col col--4')} style={{marginBottom: '1.5rem'}}>
      <Link to={link} className="featureCard" style={{display: 'block', color: 'inherit', textDecoration: 'none'}}>
        <div className="featureTitle">{title}</div>
        <p>{description}</p>
      </Link>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Bootstrap large-scale forecasting solutions on Databricks with the Many Models Forecasting Solution Accelerator.">
      <header className="heroBanner">
        <div className="container">
          <h1>Many Model Forecasting</h1>
          <p className="heroSubtitle">{siteConfig.tagline}</p>
          <div className="buttons">
            <Link className="button button--lg button--secondary" to="/docs/intro">
              Get Started
            </Link>
            <Link
              className="button button--lg button--outline button--secondary"
              href="https://github.com/databricks-industry-solutions/many-model-forecasting">
              View on GitHub
            </Link>
          </div>
        </div>
      </header>
      <main>
        <section className="features container">
          <div className="row">
            {FEATURES.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}
