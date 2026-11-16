import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { OverviewPage } from './pages/OverviewPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { LiveStreamPage } from './pages/LiveStreamPage';
import { HistoryPage } from './pages/HistoryPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { SettingsPage } from './pages/SettingsPage';
import { fetchHealth, fetchHistory, fetchModelInfo } from './services/api';
import { AnalysisResult, HealthStatus, ModelInfoResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);

  const loadInitialData = async () => {
    try {
      const [healthData, histData, infoData] = await Promise.allSettled([
        fetchHealth(),
        fetchHistory(),
        fetchModelInfo(),
      ]);

      if (healthData.status === 'fulfilled') setHealth(healthData.value);
      if (histData.status === 'fulfilled') setHistory(histData.value);
      if (infoData.status === 'fulfilled') setModelInfo(infoData.value);
    } catch (e) {
      console.error('Failed to load initial data:', e);
    }
  };

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => {});
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalysisComplete = (newResult: AnalysisResult) => {
    setHistory((prev) => [newResult, ...prev]);
  };

  const handleSelectResult = (result: AnalysisResult) => {
    setCurrentResult(result);
    setActiveTab('analyze');
  };

  const getPageTitle = () => {
    switch (activeTab) {
      case 'overview':
        return { title: 'Overview Dashboard', subtitle: 'Live Security Metrics & Activity' };
      case 'analyze':
        return { title: 'Audio Deepfake Analysis', subtitle: 'AASIST Model Inference & Diagnostics' };
      case 'stream':
        return { title: 'Real-Time Call Detection', subtitle: 'Live Audio Streaming & Fraud Mitigation' };
      case 'history':
        return { title: 'Detection Audit History', subtitle: 'Inspection Records & Export' };
      case 'model':
        return { title: 'AASIST Architecture & Limitations', subtitle: 'Empirical Specifications & Benchmark Insights' };
      case 'settings':
        return { title: 'System & Hyperparameter Settings', subtitle: 'Thresholds & Analysis Configurations' };
      default:
        return { title: 'VoiceShield', subtitle: 'AI Voice Deepfake Detection' };
    }
  };

  const pageMeta = getPageTitle();

  return (
    <div className="flex min-h-screen bg-[#0a0d14] text-slate-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendHealthy={health?.model_loaded ?? false}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          title={pageMeta.title}
          subtitle={pageMeta.subtitle}
          health={health}
          onRefreshHealth={loadInitialData}
        />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'overview' && (
            <OverviewPage
              history={history}
              onNavigateToAnalyze={() => setActiveTab('analyze')}
              onNavigateToStream={() => setActiveTab('stream')}
              onSelectResult={handleSelectResult}
            />
          )}

          {activeTab === 'analyze' && (
            <AnalysisPage
              currentResult={currentResult}
              setCurrentResult={setCurrentResult}
              onAnalysisComplete={handleAnalysisComplete}
            />
          )}

          {activeTab === 'stream' && <LiveStreamPage />}

          {activeTab === 'history' && (
            <HistoryPage
              history={history}
              onHistoryCleared={() => setHistory([])}
              onSelectResult={handleSelectResult}
            />
          )}

          {activeTab === 'model' && <ModelInfoPage modelInfo={modelInfo} />}

          {activeTab === 'settings' && <SettingsPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
