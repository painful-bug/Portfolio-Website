import { useState, useEffect, useCallback, useRef } from 'react';
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
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') || 'light';
    }
    return 'light';
  });
  const [revealing, setRevealing] = useState(false);
  const revealRef = useRef(null);

  // Apply theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Lock body scroll when modal is open
  useEffect(() => {
    if (selectedProject) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [selectedProject]);

  const toggleTheme = useCallback((e) => {
    const nextTheme = theme === 'light' ? 'dark' : 'light';

    // Get click position for the reveal origin
    const rect = e.currentTarget.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    // Set the reveal overlay color and position
    if (revealRef.current) {
      revealRef.current.style.setProperty('--reveal-x', `${x}px`);
      revealRef.current.style.setProperty('--reveal-y', `${y}px`);
      revealRef.current.style.backgroundColor =
        nextTheme === 'dark' ? '#0c1410' : '#f4f7f5';
    }

    setRevealing(true);

    // Halfway through animation, swap the actual theme
    setTimeout(() => {
      setTheme(nextTheme);
      localStorage.setItem('theme', nextTheme);
    }, 400);

    // After animation completes, hide overlay
    setTimeout(() => {
      setRevealing(false);
    }, 850);
  }, [theme]);

  return (
    <>
      {/* Dark Mode Reveal Overlay */}
      <div
        ref={revealRef}
        className={`theme-reveal ${revealing ? 'theme-reveal--active' : ''}`}
      />

      {/* Global Noise Overlay */}
      <div className="noise-overlay bg-noise" aria-hidden="true" />

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

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
