import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Home } from './components/Home';
import { DataExplorer } from './components/DataExplorer';
import { DescriptiveStats } from './components/DescriptiveStats';
import { InferenceTests } from './components/InferenceTests';
import { Modeling } from './components/Modeling';
import { Reports } from './components/Reports';
import { AIAssistant } from './components/AIAssistant';
import { DataCleaning } from './components/DataCleaning';
import { RateAdjustment } from './components/RateAdjustment';
import { Table1 } from './components/Table1';
import { Multivariate } from './components/Multivariate';
import { Survival } from './components/Survival';
import { Psychometrics } from './components/Psychometrics';
import { Associations } from './components/Associations';
import { Agreement } from './components/Agreement';
import { Diagnostics } from './components/Diagnostics';
import { GraphicsSuite } from './components/GraphicsSuite';

export default function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const renderSection = () => {
    switch (activeSection) {
      case 'home':
        return <Home />;
      case 'reports':
        return <Reports />;
      case 'ai':
        return <AIAssistant />;
      case 'cleaning':
        return <DataCleaning />;
      case 'explorer':
        return <DataExplorer />;
      case 'descriptive':
        return <DescriptiveStats />;
      case 'rates':
        return <RateAdjustment />;
      case 'table1':
        return <Table1 />;
      case 'inference':
        return <InferenceTests />;
      case 'modeling':
        return <Modeling />;
      case 'multivariate':
        return <Multivariate />;
      case 'survival':
        return <Survival />;
      case 'psychometrics':
        return <Psychometrics />;
      case 'associations':
        return <Associations />;
      case 'agreement':
        return <Agreement />;
      case 'diagnostics':
        return <Diagnostics />;
      case 'graphics':
        return <GraphicsSuite />;
      default:
        return <Home />;
    }
  };

  return (
    <div className="flex h-screen bg-[#f8f9fb] overflow-hidden">
      <Sidebar 
        activeSection={activeSection} 
        setActiveSection={setActiveSection}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
      />
      <div className={`flex-1 flex flex-col transition-all duration-300 ${sidebarCollapsed ? 'ml-20' : 'ml-64'}`}>
        <Header activeSection={activeSection} />
        <main className="flex-1 overflow-auto p-8">
          {renderSection()}
        </main>
      </div>
    </div>
  );
}
