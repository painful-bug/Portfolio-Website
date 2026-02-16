export default function WorkGrid({ projects, onProjectClick }) {
  return (
    <section className="workgrid" id="work">
      <div className="workgrid__inner">
        {/* Sticky Sidebar */}
        <aside className="workgrid__sidebar">
          <div className="workgrid__sidebar-sticky">
            <div className="workgrid__sidebar-accent" />
            <h2 className="workgrid__sidebar-title font-serif">
              Selected <br className="workgrid__br-lg" />Works
            </h2>
            <span className="workgrid__sidebar-count">
              ({String(projects.length).padStart(2, '0')} Projects)
            </span>
            <p className="workgrid__sidebar-desc">
              A curated collection of digital artifacts, exploring the
              intersection of finance, art, and utility.
            </p>
          </div>
        </aside>

        {/* Cards Grid */}
        <div className="workgrid__grid">
          {projects.map((project) => (
            <article
              key={project.id}
              className={`project-card hover-lift glass-card ${
                project.span === 'full' ? 'workgrid__card--full' : ''
              } ${project.stagger ? 'workgrid__card--stagger' : ''}`}
              onClick={() => onProjectClick(project)}
            >
              {/* Hover Badge */}
              <div className="project-card__badge">
                <div className="project-card__badge-inner">
                  <span className="project-card__badge-text">View Case</span>
                  <span className="material-symbols-outlined project-card__badge-icon">
                    arrow_outward
                  </span>
                </div>
              </div>

              {/* Image */}
              <div
                className={`project-card__image-wrap ${
                  project.span === 'full'
                    ? 'project-card__image-wrap--tall'
                    : ''
                }`}
              >
                <img
                  src={project.image}
                  alt={project.title}
                  className="project-card__image mist-image"
                  loading="lazy"
                />
                <div className="project-card__image-noise bg-noise" />
              </div>

              {/* Content */}
              <div className="project-card__content">
                <div className="project-card__info">
                  <h3 className="project-card__title font-serif">
                    {project.title}
                  </h3>
                  <p className="project-card__subtitle">{project.subtitle}</p>
                </div>
                <div className="project-card__tags">
                  {project.tags.map((tag) => (
                    <span key={tag} className="project-card__tag">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>

      {/* Bottom Fade */}
      <div className="workgrid__fade" />
    </section>
  );
}
