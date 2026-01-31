// components/panels/ModelsPanel.js
// CookieRun AI Platform - ML 모델 관리 패널

import { useEffect, useState } from 'react';
import { Brain, Layers, FlaskConical, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';

// CookieRun 샘플 모델 레지스트리
const SAMPLE_MODELS = [
  {
    name: 'cookierun-translation-model',
    description: '세계관 번역 품질 평가 모델',
    model_type: 'artifact',
    versions: [
      { version: 2, stage: 'Production', created_at: '2026-01-30' },
      { version: 1, stage: 'Archived', created_at: '2026-01-15' },
    ]
  },
  {
    name: 'cookierun-segment-model',
    description: '유저 세그먼트 분류 모델',
    model_type: 'artifact',
    versions: [
      { version: 1, stage: 'Production', created_at: '2026-01-28' },
    ]
  },
  {
    name: 'cookierun-anomaly-model',
    description: '이상 행동 유저 탐지 모델',
    model_type: 'artifact',
    versions: [
      { version: 1, stage: 'Staging', created_at: '2026-01-25' },
    ]
  },
  {
    name: 'cookierun-cookie-recommend',
    description: '쿠키 캐릭터 추천 모델',
    model_type: 'artifact',
    versions: [
      { version: 3, stage: 'Production', created_at: '2026-01-29' },
      { version: 2, stage: 'Staging', created_at: '2026-01-20' },
      { version: 1, stage: 'Archived', created_at: '2026-01-10' },
    ]
  },
];

// CookieRun 샘플 실험 데이터
const SAMPLE_EXPERIMENTS = [
  {
    experiment_id: '1',
    name: 'cookierun-ai-platform',
    lifecycle_stage: 'active',
    runs: [
      {
        run_id: 'run_001',
        run_name: 'translation_quality_v2',
        status: 'FINISHED',
        start_time: Date.now() - 86400000,
        metrics: {
          bleu_score: 0.892,
          accuracy: 0.945,
          f1_score: 0.931,
        },
        params: {
          model_type: 'transformer',
          epochs: 50,
          learning_rate: 0.0001,
        }
      },
      {
        run_id: 'run_002',
        run_name: 'segment_classifier',
        status: 'FINISHED',
        start_time: Date.now() - 172800000,
        metrics: {
          accuracy: 0.8756,
          f1_macro: 0.8421,
          precision: 0.8632,
        },
        params: {
          n_classes: 5,
          max_depth: 15,
          n_estimators: 200,
        }
      },
      {
        run_id: 'run_003',
        run_name: 'anomaly_detector',
        status: 'FINISHED',
        start_time: Date.now() - 259200000,
        metrics: {
          anomaly_ratio: 0.032,
          precision: 0.912,
          recall: 0.876,
        },
        params: {
          contamination: 0.03,
          n_estimators: 150,
          algorithm: 'isolation_forest',
        }
      },
      {
        run_id: 'run_004',
        run_name: 'cookie_recommender_v3',
        status: 'FINISHED',
        start_time: Date.now() - 345600000,
        metrics: {
          ndcg_at_10: 0.7823,
          map_at_10: 0.6541,
          hit_rate: 0.8912,
        },
        params: {
          n_cookies: 85,
          embedding_dim: 128,
          similarity: 'cosine',
        }
      },
    ]
  },
  {
    experiment_id: '2',
    name: 'cookierun-user-analysis',
    lifecycle_stage: 'active',
    runs: [
      {
        run_id: 'run_005',
        run_name: 'churn_prediction',
        status: 'FINISHED',
        start_time: Date.now() - 432000000,
        metrics: {
          auc_roc: 0.8934,
          precision: 0.8521,
          recall: 0.7892,
        },
        params: {
          model: 'xgboost',
          max_depth: 8,
          n_estimators: 300,
        }
      },
      {
        run_id: 'run_006',
        run_name: 'ltv_prediction',
        status: 'RUNNING',
        start_time: Date.now() - 3600000,
        metrics: {},
        params: {
          model: 'lightgbm',
          objective: 'regression',
        }
      },
    ]
  },
];

export default function ModelsPanel({ auth, apiCall }) {
  const [mlflowData, setMlflowData] = useState([]);
  const [registeredModels, setRegisteredModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selecting, setSelecting] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null); // 선택된 모델 상태
  const [message, setMessage] = useState(null);
  const [usingSample, setUsingSample] = useState(false);

  // 로컬스토리지에서 선택된 모델 복원
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('cookierun_selected_model');
      if (saved) {
        setSelectedModel(saved);
      }
    }
  }, []);

  // 선택된 모델 로컬스토리지에 저장
  useEffect(() => {
    if (typeof window !== 'undefined' && selectedModel) {
      localStorage.setItem('cookierun_selected_model', selectedModel);
    }
  }, [selectedModel]);

  useEffect(() => {
    async function fetchMLflowData() {
      setLoading(true);
      let gotRealData = false;

      try {
        // MLflow 실험 데이터 조회
        const expRes = await apiCall({
          endpoint: '/api/mlflow/experiments',
          auth,
          timeoutMs: 10000,
        });

        if (expRes?.status === 'SUCCESS' && expRes.data?.length > 0) {
          // CookieRun 관련 실험만 필터링
          const cookierunExps = expRes.data.filter(exp =>
            exp.name.toLowerCase().includes('cookie') ||
            exp.name.toLowerCase().includes('translation') ||
            exp.name.toLowerCase().includes('segment') ||
            exp.name.toLowerCase().includes('anomaly')
          );

          if (cookierunExps.length > 0) {
            setMlflowData(cookierunExps);
            gotRealData = true;
          }
        }
      } catch (e) {
        console.log('MLflow 실험 API fallback');
      }

      try {
        // MLflow 모델 레지스트리 조회
        const modelsRes = await apiCall({
          endpoint: '/api/mlflow/models',
          auth,
          timeoutMs: 10000,
        });

        if (modelsRes?.status === 'SUCCESS' && modelsRes.data?.length > 0) {
          // CookieRun 관련 모델만 필터링
          const cookierunModels = modelsRes.data.filter(m =>
            m.name.toLowerCase().includes('cookie') ||
            m.name.toLowerCase().includes('translation') ||
            m.name.toLowerCase().includes('segment') ||
            m.name.toLowerCase().includes('anomaly')
          );

          if (cookierunModels.length > 0) {
            setRegisteredModels(cookierunModels);
            gotRealData = true;
          }
        }
      } catch (e) {
        console.log('MLflow 모델 API fallback');
      }

      // CookieRun 데이터가 없으면 샘플 데이터 사용
      if (!gotRealData) {
        setMlflowData(SAMPLE_EXPERIMENTS);
        setRegisteredModels(SAMPLE_MODELS);
        setUsingSample(true);
      } else {
        setUsingSample(false);
      }

      setLoading(false);
    }

    if (auth) {
      fetchMLflowData();
    }
  }, [auth, apiCall]);

  const formatTimestamp = (ts) => {
    if (!ts) return '-';
    const date = new Date(ts);
    return date.toLocaleString('ko-KR');
  };

  const handleSelectModel = async (modelName, version) => {
    const modelKey = `${modelName}-${version}`;
    setSelecting(modelKey);
    setMessage(null);

    try {
      const res = await apiCall({
        endpoint: '/api/mlflow/models/select',
        auth,
        method: 'POST',
        data: { model_name: modelName, version: String(version) },
        timeoutMs: 30000,
      });

      setSelecting(null);

      if (res?.status === 'SUCCESS') {
        setSelectedModel(modelKey);
        setMessage({ type: 'success', text: res.message || `${modelName} v${version} 모델이 로드되었습니다` });
      } else {
        setMessage({ type: 'error', text: res?.error || '모델 로드 실패' });
      }
    } catch (e) {
      setSelecting(null);
      // 데모 모드에서도 선택 상태 반영
      setSelectedModel(modelKey);
      setMessage({ type: 'success', text: `${modelName} v${version} 모델이 선택되었습니다 (데모)` });
    }

    setTimeout(() => setMessage(null), 5000);
  };

  const handleRefresh = async () => {
    setMlflowData([]);
    setRegisteredModels([]);
    setLoading(true);
    let gotRealData = false;

    try {
      const expRes = await apiCall({
        endpoint: '/api/mlflow/experiments',
        auth,
        timeoutMs: 10000,
      });

      if (expRes?.status === 'SUCCESS' && expRes.data?.length > 0) {
        const cookierunExps = expRes.data.filter(exp =>
          exp.name.toLowerCase().includes('cookie') ||
          exp.name.toLowerCase().includes('translation') ||
          exp.name.toLowerCase().includes('segment') ||
          exp.name.toLowerCase().includes('anomaly')
        );
        if (cookierunExps.length > 0) {
          setMlflowData(cookierunExps);
          gotRealData = true;
        }
      }
    } catch (e) {
      console.log('MLflow experiments refresh fallback');
    }

    try {
      const modelsRes = await apiCall({
        endpoint: '/api/mlflow/models',
        auth,
        timeoutMs: 10000,
      });

      if (modelsRes?.status === 'SUCCESS' && modelsRes.data?.length > 0) {
        const cookierunModels = modelsRes.data.filter(m =>
          m.name.toLowerCase().includes('cookie') ||
          m.name.toLowerCase().includes('translation') ||
          m.name.toLowerCase().includes('segment') ||
          m.name.toLowerCase().includes('anomaly')
        );
        if (cookierunModels.length > 0) {
          setRegisteredModels(cookierunModels);
          gotRealData = true;
        }
      }
    } catch (e) {
      console.log('MLflow models refresh fallback');
    }

    if (!gotRealData) {
      setMlflowData(SAMPLE_EXPERIMENTS);
      setRegisteredModels(SAMPLE_MODELS);
      setUsingSample(true);
    } else {
      setUsingSample(false);
    }

    setLoading(false);
  };

  return (
    <div>
      <SectionHeader
        title="CookieRun ML 모델 관리"
        subtitle="번역 · 세그먼트 · 추천 · 이상탐지 모델"
        right={
          <div className="flex items-center gap-2">
            <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
              usingSample
                ? 'border-amber-400/50 bg-amber-50 text-amber-700'
                : 'border-green-400/50 bg-green-50 text-green-700'
            }`}>
              {usingSample ? 'SAMPLE' : 'LIVE'}
            </span>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="rounded-full border-2 border-cookie-orange/20 bg-white/80 p-1.5 hover:bg-cookie-beige transition disabled:opacity-50"
            >
              <RefreshCw size={14} className={`text-cookie-brown ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        }
      />

      {message && (
        <div className={`mb-4 p-3 rounded-2xl flex items-center gap-2 text-sm ${
          message.type === 'success'
            ? 'bg-green-50 border-2 border-green-200 text-green-700'
            : 'bg-red-50 border-2 border-red-200 text-red-700'
        }`}>
          {message.type === 'success' ? <CheckCircle size={16} /> : <XCircle size={16} />}
          {message.text}
        </div>
      )}

      {/* Model Registry Section */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Layers size={18} className="text-cookie-orange" />
          <h3 className="text-sm font-black text-cookie-brown">Model Registry</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {registeredModels.map((model) => (
            <div key={model.name} className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Brain size={16} className="text-cookie-orange" />
                  <span className="font-bold text-cookie-brown text-sm">{model.name}</span>
                </div>
                {model.model_type === 'artifact' && (
                  <span className="text-[10px] px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full font-bold">
                    Artifact
                  </span>
                )}
              </div>
              <p className="text-xs text-cookie-brown/60 mb-3">
                {model.description || '설명 없음'}
              </p>
              <div className="space-y-2">
                {model.versions.map((v) => {
                  const modelKey = `${model.name}-${v.version}`;
                  const isSelected = selectedModel === modelKey;
                  const isSelecting = selecting === modelKey;

                  return (
                    <div key={v.version} className={`flex items-center justify-between p-2.5 rounded-xl transition ${
                      isSelected
                        ? 'bg-gradient-to-r from-cookie-yellow/30 to-cookie-orange/20 border-2 border-cookie-orange'
                        : 'bg-cookie-beige/50'
                    }`}>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-cookie-brown">v{v.version}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                          v.stage === 'Production' ? 'bg-green-100 text-green-700' :
                          v.stage === 'Staging' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-500'
                        }`}>
                          {v.stage || 'None'}
                        </span>
                        {isSelected && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-cookie-orange text-white">
                            ✓ 사용중
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => handleSelectModel(model.name, v.version)}
                        disabled={isSelecting || isSelected}
                        className={`text-xs px-3 py-1.5 rounded-lg font-bold shadow transition ${
                          isSelected
                            ? 'bg-gray-200 text-gray-500 cursor-default shadow-none'
                            : 'bg-gradient-to-r from-cookie-yellow to-cookie-orange text-white hover:shadow-md'
                        } disabled:opacity-50`}
                      >
                        {isSelecting ? '로딩...' : isSelected ? '선택됨' : '선택'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Experiments Section */}
      <div className="flex items-center gap-2 mb-4">
        <FlaskConical size={18} className="text-cookie-orange" />
        <h3 className="text-sm font-black text-cookie-brown">실험 기록</h3>
      </div>

      {mlflowData.length ? mlflowData.map((exp) => (
        <div key={exp.experiment_id} className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur mb-4">
          <div className="flex justify-between items-center mb-4">
            <span className="font-bold text-cookie-brown">{exp.name}</span>
            <span className={`text-[10px] px-2 py-1 rounded-full font-bold ${
              exp.lifecycle_stage === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
            }`}>
              {exp.lifecycle_stage}
            </span>
          </div>

          {exp.runs && exp.runs.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cookie-orange/10">
                    <th className="text-left py-2 px-2 text-cookie-brown font-bold text-xs">Run Name</th>
                    <th className="text-left py-2 px-2 text-cookie-brown font-bold text-xs">Status</th>
                    <th className="text-left py-2 px-2 text-cookie-brown font-bold text-xs">시작 시간</th>
                    <th className="text-left py-2 px-2 text-cookie-brown font-bold text-xs">Metrics</th>
                    <th className="text-left py-2 px-2 text-cookie-brown font-bold text-xs">Params</th>
                  </tr>
                </thead>
                <tbody>
                  {exp.runs.map((run) => (
                    <tr key={run.run_id} className="border-b border-cookie-orange/5 hover:bg-cookie-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cookie-brown">
                        {run.run_name || run.run_id.slice(0, 8)}
                      </td>
                      <td className="py-3 px-2">
                        <span className={`text-[10px] px-2 py-1 rounded-full font-bold ${
                          run.status === 'FINISHED' ? 'bg-green-100 text-green-700' :
                          run.status === 'RUNNING' ? 'bg-blue-100 text-blue-700 animate-pulse' :
                          run.status === 'FAILED' ? 'bg-red-100 text-red-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="py-3 px-2 text-cookie-brown/60 text-xs">
                        {formatTimestamp(run.start_time)}
                      </td>
                      <td className="py-3 px-2">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(run.metrics || {}).map(([k, v]) => (
                            <span key={k} className="text-[10px] bg-cookie-yellow/30 text-cookie-brown px-2 py-0.5 rounded-full font-semibold">
                              {k}: {typeof v === 'number' ? v.toFixed(4) : v}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="py-3 px-2">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(run.params || {}).slice(0, 3).map(([k, v]) => (
                            <span key={k} className="text-[10px] bg-cookie-beige text-cookie-brown/70 px-2 py-0.5 rounded-full">
                              {k}: {v}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-cookie-brown/50 py-4 text-center">
              실험 기록이 없습니다.
            </div>
          )}
        </div>
      )) : (
        <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-8 text-center">
          <FlaskConical size={32} className="mx-auto mb-3 text-cookie-brown/30" />
          <p className="text-sm text-cookie-brown/60">MLflow 실험이 없습니다.</p>
          <p className="text-xs text-cookie-brown/40 mt-1">
            노트북을 실행하여 모델을 학습하세요.
          </p>
        </div>
      )}
    </div>
  );
}
