import { useEffect, useRef } from 'react';

export default function ArtifactModal({ project, onClose }) {
  const backdropRef = useRef(null);

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const handleBackdropClick = (e) => {
    if (e.target === backdropRef.current) onClose();
  };

  return (
    <div
      className="artifact-backdrop"
      ref={backdropRef}
      onClick={handleBackdropClick}
    >
      <div className="artifact animate-swoop-in">
        {/* Noise Texture */}
        <div className="artifact__noise bg-noise" aria-hidden="true" />

        {/* Close Button */}
        <div className="artifact__controls">
          <button className="artifact__close" onClick={onClose} aria-label="Close">
            <span className="material-symbols-outlined artifact__close-icon">
              close
            </span>
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="artifact__scroll custom-scrollbar">
          {/* Hero Image */}
          <div className="artifact__hero">
            <img
              src={project.heroImage || project.image}
              alt={project.title}
              className="artifact__hero-img"
            />
            <div className="artifact__hero-overlay" />
            <div className="artifact__hero-title-wrap">
              <h1 className="artifact__hero-title font-serif">
                {project.title}
              </h1>
            </div>
          </div>

          {/* Two-Column Editorial Layout */}
          <div className="artifact__body">
            {/* Left: Metadata & Visuals */}
            <div className="artifact__meta-col">
              <div className="artifact__meta-group">
                <div className="artifact__meta-block">
                  <p className="artifact__meta-label">Role</p>
                  <p className="artifact__meta-value font-serif">
                    {project.role}
                  </p>
                </div>
                <div className="artifact__meta-block">
                  <p className="artifact__meta-label">Year</p>
                  <p className="artifact__meta-value--plain">{project.year}</p>
                </div>
                <div className="artifact__meta-block">
                  <p className="artifact__meta-label">Tech Stack</p>
                  <div className="artifact__tech-tags">
                    {project.techStack.map((tech) => (
                      <span key={tech} className="artifact__tech-tag font-mono">
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Secondary Image */}
              {project.detailImage && (
                <div className="artifact__detail-image-wrap">
                  <img
                    src={project.detailImage}
                    alt={`${project.title} detail`}
                    className="artifact__detail-image"
                  />
                </div>
              )}

              {/* Visit Link */}
              <a href="#" className="artifact__visit-btn">
                <span className="artifact__visit-text font-serif">
                  Visit Live Site
                </span>
                <span className="material-symbols-outlined">arrow_outward</span>
              </a>
            </div>

            {/* Right: Narrative */}
            <div className="artifact__narrative-col">
              <div className="artifact__article-header">
                <h2 className="artifact__article-title font-serif">
                  {project.articleTitle || 'Case Study'}
                </h2>
                <div className="artifact__article-underline" />
              </div>

              {/* Description with Drop Cap */}
              <div className="artifact__prose">
                {project.description.map((para, i) => (
                  <p
                    key={i}
                    className={`artifact__paragraph ${
                      i === 0 ? 'drop-cap' : ''
                    }`}
                  >
                    {para}
                  </p>
                ))}
              </div>

              {/* Technical Approach */}
              <div className="artifact__technical">
                <h3 className="artifact__technical-title font-serif">
                  The Technical Approach
                </h3>
                <p className="artifact__paragraph">{project.technicalApproach}</p>

                {/* Code Snippet */}
                {project.codeSnippet && (
                  <div className="artifact__code-block">
                    <div className="artifact__code-dots">
                      <div className="artifact__code-dot artifact__code-dot--red" />
                      <div className="artifact__code-dot artifact__code-dot--yellow" />
                      <div className="artifact__code-dot artifact__code-dot--green" />
                    </div>
                    <code className="artifact__code font-mono">
                      {project.codeSnippet}
                    </code>
                  </div>
                )}
              </div>

              {/* Footer Credits */}
              <div className="artifact__credits">
                <span>Aishik Bandyopadhyay</span>
                <span className="artifact__credits-dot" />
                <span>Portfolio 2025</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
