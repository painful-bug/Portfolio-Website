import { useEffect, useRef } from 'react';

export default function FogWalk() {
  const titleRef = useRef(null);
  const statusRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      if (titleRef.current) {
        titleRef.current.style.transform = `translateY(${scrollY * 0.15}px)`;
      }
      if (statusRef.current) {
        statusRef.current.style.transform = `translateY(${scrollY * 0.05}px)`;
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <section className="fogwalk">
      {/* Mist Layers */}
      <div className="fogwalk__mist" aria-hidden="true">
        <div className="fogwalk__mist-blob fogwalk__mist-blob--top" />
        <div className="fogwalk__mist-blob fogwalk__mist-blob--bottom" />
        <div className="fogwalk__mist-noise bg-noise animate-drift" />
      </div>

      {/* Content */}
      <div className="fogwalk__content">
        {/* Left Column */}
        <div className="fogwalk__left">
          {/* Status Indicator */}
          <div
            ref={statusRef}
            className="fogwalk__status animate-fade-in-blur delay-1"
          >
            <span className="fogwalk__status-dot-wrap">
              <span className="fogwalk__status-dot-ping" />
              <span className="fogwalk__status-dot" />
            </span>
            <span className="fogwalk__status-text">
              Currently exploring AI &amp; Data Engineering
            </span>
          </div>

          {/* Hero Title */}
          <h1
            ref={titleRef}
            className="fogwalk__title animate-fade-in-blur delay-2"
          >
            Aishik Bandyopadhyay<span className="fogwalk__title-dot">.</span>
          </h1>

          {/* Subtitle & Description */}
          <div className="fogwalk__subtitle-group animate-fade-in-blur delay-3">
            <h2 className="fogwalk__subtitle">
              Data Scientist &amp;<br />AI Engineer
            </h2>
            <p className="fogwalk__description">
              Building intelligent systems from raw data. I craft end-to-end ML pipelines,
              agentic AI applications, and data architectures that turn chaos into clarity.
            </p>
          </div>

          {/* CTA */}
          <div className="fogwalk__cta animate-fade-in-blur delay-3">
            <a href="#work" className="fogwalk__cta-btn">
              <span className="fogwalk__cta-text">Explore Selected Works</span>
              <span className="material-symbols-outlined fogwalk__cta-arrow">
                arrow_forward
              </span>
            </a>
          </div>
        </div>

        {/* Right Column – Decorative meta */}
        <div className="fogwalk__right animate-fade-in-blur delay-3">
          <span className="fogwalk__meta">IIT Madras &middot; BS Data Science</span>
          <div className="fogwalk__meta-line" />
          <span className="fogwalk__meta">Kolkata, India</span>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="fogwalk__scroll animate-fade-in-blur delay-3">
        <span className="fogwalk__scroll-label">Scroll</span>
        <div className="fogwalk__scroll-line" />
      </div>

      {/* Decorative Floating Element */}
      <div className="fogwalk__decor animate-fade-in-blur delay-2" aria-hidden="true">
        <div className="fogwalk__decor-box" />
      </div>
    </section>
  );
}
