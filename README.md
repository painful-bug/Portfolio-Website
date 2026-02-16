# Aishik Bandyopadhyay — Data Scientist & AI Engineer Portfolio

[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-7.3-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy to GitHub Pages](https://github.com/painful-bug/Portfolio-Website/actions/workflows/deploy.yml/badge.svg)](https://github.com/painful-bug/Portfolio-Website/actions/workflows/deploy.yml)

A premium, narrative-driven portfolio website showcasing engineering excellence in Data Science, AI, and Software Development. Designed with a focus on rich aesthetics, smooth interactions, and a clean user experience.

[**Live Demo »**](https://painful-bug.github.io/Portfolio-Website/)

---

## ✨ Features

- **Premium Design Aesthetics**: Custom-built UI with a focus on typography, whitespace, and a subtle "noise" texture for a high-end feel.
- **Dynamic Hero Background**: A custom "FogWalk" animation creating a sophisticated atmosphere.
- **Circular Theme Reveal**: A bespoke Dark/Light mode transition using a circular expand/contract animation that originates from the click position.
- **Narrative Project Cards**: Instead of just titles, each project is presented as an "artifact" with deep technical write-ups, code snippets, and structured metadata (Role, Tech Stack, Methodology).
- **Responsive Artifact Modals**: Detailed project views with smooth scroll-locking and elegant transitions.
- **Integrated Contact System**: A functional, spam-protected contact form powered by FormSubmit.co.
- **Automated CI/CD**: Seamless deployment to GitHub Pages via specialized GitHub Actions workflows.

## 🛠️ Tech Stack

- **Frontend**: [React 19](https://react.dev/) for component-based UI.
- **Build Tool**: [Vite](https://vitejs.dev/) for lightning-fast bundling and development.
- **Styling**: Vanilla CSS with modern features (CSS Variables, Flexbox/Grid, Keyframe Animations).
- **Icons**: [Google Material Symbols](https://fonts.google.com/icons).
- **Fonts**: Inter and Serif combinations for a journalistic, professional aesthetic.

## 📂 Project Structure

```text
├── .github/workflows/  # CI/CD Deployment pipeline
├── public/             # Static assets (Favicons, CV)
├── src/
│   ├── components/     # React components and their CSS
│   │   ├── FogWalk.jsx  # Hero animation
│   │   ├── WorkGrid.jsx # Project gallery
│   │   └── ArtifactModal.jsx # Detail view
│   ├── data/           # Project content and metadata
│   ├── App.jsx         # Main application logic & Theme engine
│   ├── index.css       # Global design system & layout
│   └── main.jsx        # Entry point
```

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v20 or higher recommended)
- [npm](https://www.npmjs.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/painful-bug/Portfolio-Website.git
   cd Portfolio-Website
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## 🚢 Deployment

The project is configured to automatically deploy to GitHub Pages whenever changes are pushed to the `main` branch.

- **Workflow**: `.github/workflows/deploy.yml`
- **Trigger**: `push` to `main` branch.
- **Permissions**: Pages write access, ID token write access.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Designed with ❤️ by Aishik Bandyopadhyay
