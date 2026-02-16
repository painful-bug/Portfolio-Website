import { useState, useEffect } from 'react';
import './components/Navbar.css';
import './components/FogWalk.css';
import './components/About.css';
import './components/WorkGrid.css';
import './components/ArtifactModal.css';
import './components/Signal.css';

import Navbar from './components/Navbar';
import FogWalk from './components/FogWalk';
import About from './components/About';
import WorkGrid from './components/WorkGrid';
import ArtifactModal from './components/ArtifactModal';
import Signal from './components/Signal';
import projects from './data/projectsData';

function App() {
  const [selectedProject, setSelectedProject] = useState(null);

  /* Lock body scroll when modal is open */
  useEffect(() => {
    if (selectedProject) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [selectedProject]);

  return (
    <>
      {/* Global Noise Overlay */}
      <div className="noise-overlay bg-noise" aria-hidden="true" />

      <Navbar />

      <main>
        <FogWalk />
        <About />
        <WorkGrid
          projects={projects}
          onProjectClick={setSelectedProject}
        />
      </main>

      <Signal />

      {selectedProject && (
        <ArtifactModal
          project={selectedProject}
          onClose={() => setSelectedProject(null)}
        />
      )}
    </>
  );
}

export default App;
