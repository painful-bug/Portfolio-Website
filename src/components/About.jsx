export default function About() {
  return (
    <section className="about" id="about">
      <div className="about__inner">
        {/* Section Label */}
        <div className="about__label animate-fade-in-blur delay-1">
          <div className="about__label-line" />
          <span className="about__label-text">About</span>
        </div>

        {/* Main Content Grid */}
        <div className="about__grid">
          {/* Left: Bio */}
          <div className="about__bio">
            <h2 className="about__heading font-serif">
              Building intelligent systems<br />
              <span className="about__heading-accent">from raw data.</span>
            </h2>
            <p className="about__text">
              I&apos;m a Data Science student at <strong>IIT Madras</strong> and Computer Science
              undergraduate at <strong>Techno International New Town</strong>, Kolkata.
              I specialize in building end-to-end ML pipelines, data warehouses,
              and AI-powered applications that bridge the gap between raw data and
              real-world impact.
            </p>
            <p className="about__text about__text--secondary">
              My work spans deep learning architectures, agentic AI systems with
              self-correction capabilities, and production-grade data engineering
              with dimensional modeling. I believe the best systems are the ones
              that make complexity invisible.
            </p>
          </div>

          {/* Right: Credentials */}
          <div className="about__credentials">
            {/* Education */}
            <div className="about__section">
              <h3 className="about__section-title">Education</h3>
              <div className="about__edu-list">
                <div className="about__edu-item">
                  <div className="about__edu-header">
                    <span className="about__edu-school">IIT Madras</span>
                    <span className="about__edu-year">2027</span>
                  </div>
                  <span className="about__edu-degree">BS in Data Science — CGPA: 7.98</span>
                </div>
                <div className="about__edu-item">
                  <div className="about__edu-header">
                    <span className="about__edu-school">Techno International New Town</span>
                    <span className="about__edu-year">2027</span>
                  </div>
                  <span className="about__edu-degree">BTech in Computer Science — CGPA: 8.33</span>
                </div>
              </div>
            </div>

            {/* Technical Skills */}
            <div className="about__section">
              <h3 className="about__section-title">Technical Arsenal</h3>
              <div className="about__skills">
                <div className="about__skill-row">
                  <span className="about__skill-label">Languages</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">Python</span>
                    <span className="about__skill-tag">SQL</span>
                    <span className="about__skill-tag">JavaScript</span>
                  </div>
                </div>
                <div className="about__skill-row">
                  <span className="about__skill-label">ML & AI</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">TensorFlow</span>
                    <span className="about__skill-tag">PyTorch</span>
                    <span className="about__skill-tag">Deep Learning</span>
                    <span className="about__skill-tag">Agentic AI</span>
                  </div>
                </div>
                <div className="about__skill-row">
                  <span className="about__skill-label">Data Eng.</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">MS SQL Server</span>
                    <span className="about__skill-tag">ETL</span>
                    <span className="about__skill-tag">Star Schema</span>
                    <span className="about__skill-tag">Medallion Arch.</span>
                  </div>
                </div>
                <div className="about__skill-row">
                  <span className="about__skill-label">Web Dev</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">ReactJS</span>
                    <span className="about__skill-tag">VueJS</span>
                    <span className="about__skill-tag">Flask</span>
                    <span className="about__skill-tag">FastAPI</span>
                  </div>
                </div>
                <div className="about__skill-row">
                  <span className="about__skill-label">Tools</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">Tableau</span>
                    <span className="about__skill-tag">Git</span>
                    <span className="about__skill-tag">Celery</span>
                    <span className="about__skill-tag">SQLite</span>
                  </div>
                </div>
                <div className="about__skill-row">
                  <span className="about__skill-label">Math</span>
                  <div className="about__skill-tags">
                    <span className="about__skill-tag">Linear Algebra</span>
                    <span className="about__skill-tag">Probability</span>
                    <span className="about__skill-tag">Multivariate Calculus</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
