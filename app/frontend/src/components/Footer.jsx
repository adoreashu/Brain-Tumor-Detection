const Footer = () => {
  return (
    <footer className="glass-panel mt-12 py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center text-sm text-[var(--text-secondary)]">
          <p>&copy; {new Date().getFullYear()} Brain Tumour Detection System by Ashu.</p>
          <p style={{ marginTop: '0.5rem', color: 'var(--color-error)' }}>
            Disclaimer: This tool is for educational and research purposes only. Do not use for clinical diagnosis.
          </p>
          <div className="footer-links">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="/about">About the Model</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
