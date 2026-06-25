import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DataProvider } from './context/DataContext.jsx'
import Rail from './components/Rail.jsx'
import Footer from './components/Footer.jsx'
import Overview from './pages/Overview.jsx'
import Methods from './pages/Methods.jsx'
import Results from './pages/Results.jsx'
import Q2Alternatives from './pages/Q2Alternatives.jsx'
import Q3Zeroshot from './pages/Q3Zeroshot.jsx'
import Q6Ablations from './pages/Q6Ablations.jsx'
import CaseStudies from './pages/CaseStudies.jsx'
import Report from './pages/Report.jsx'
import Evidence from './pages/Evidence.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <DataProvider>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <Rail />
        <main id="main-content" className="site-main">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/methods" element={<Methods />} />
            <Route path="/results" element={<Results />} />
            <Route path="/q2-alternatives" element={<Q2Alternatives />} />
            <Route path="/q3-zeroshot" element={<Q3Zeroshot />} />
            <Route path="/q6-ablations" element={<Q6Ablations />} />
            <Route path="/case-studies" element={<CaseStudies />} />
            <Route path="/report" element={<Report />} />
            <Route path="/evidence" element={<Evidence />} />
          </Routes>
        </main>
        <Footer />
      </DataProvider>
    </BrowserRouter>
  )
}
