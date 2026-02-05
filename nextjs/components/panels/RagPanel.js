// components/panels/RagPanel.js
import { useCallback, useEffect, useState } from 'react';
import { Upload, FileText, Trash2, RefreshCw, CheckCircle, XCircle, AlertCircle, Image, ScanText, Zap, GitBranch, Search, Loader2, CheckSquare, Square, Sparkles } from 'lucide-react';
import SectionHeader from '../SectionHeader';

export default function RagPanel({ auth, apiCall, addLog, settings, setSettings }) {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedOcrFile, setSelectedOcrFile] = useState(null);
  const [ocrUploading, setOcrUploading] = useState(false);
  const [ocrResult, setOcrResult] = useState(null);
  const [lightragBuilding, setLightragBuilding] = useState(false);
  const [lightragStatus, setLightragStatus] = useState(null);
  const [showReindexTooltip, setShowReindexTooltip] = useState(false);
  const [showUploadTooltip, setShowUploadTooltip] = useState(false);
  const [showOcrTooltip, setShowOcrTooltip] = useState(false);
  const [showDeleteTooltip, setShowDeleteTooltip] = useState(false);
  const [selectedForDelete, setSelectedForDelete] = useState(new Set());
  const [deleting, setDeleting] = useState(false);

  // RAG 상태 로드
  const loadStatus = useCallback(async () => {
    if (!auth) return;
    setLoading(true);

    try {
      const res = await apiCall({
        endpoint: '/api/rag/status',
        method: 'GET',
        auth,
      });

      if (res?.status === 'SUCCESS') {
        setStatus(res);
      }

      // LightRAG 상태도 로드
      const lightragRes = await apiCall({
        endpoint: '/api/lightrag/status',
        method: 'GET',
        auth,
      });

      if (lightragRes?.status === 'SUCCESS') {
        setLightragStatus(lightragRes);
      }
    } catch (e) {
      console.error('RAG 상태 로드 실패:', e);
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth]);

  // 파일 목록 로드
  const loadFiles = useCallback(async () => {
    if (!auth) return;

    try {
      const res = await apiCall({
        endpoint: '/api/rag/files',
        method: 'GET',
        auth,
      });

      if (res?.status === 'SUCCESS' && Array.isArray(res.files)) {
        setFiles(res.files);
      }
    } catch (e) {
      console.error('파일 목록 로드 실패:', e);
    }
  }, [apiCall, auth]);

  // 초기 로드
  useEffect(() => {
    loadStatus();
    loadFiles();
  }, [loadStatus, loadFiles]);

  // 파일 업로드 (다중 파일 지원 - 배치 업로드 후 한 번만 재빌드)
  const handleFileUpload = useCallback(async () => {
    if (selectedFiles.length === 0 || !auth) return;

    setUploading(true);
    const totalFiles = selectedFiles.length;
    const results = { success: [], failed: [] };

    // 1단계: 모든 파일 업로드 (skip_reindex=true로 재빌드 건너뛰기)
    for (let i = 0; i < totalFiles; i++) {
      const file = selectedFiles[i];
      addLog?.('RAG 문서 업로드', `${file.name} (${i + 1}/${totalFiles})`);

      try {
        const formData = new FormData();
        formData.append('file', file);

        const base = process.env.NEXT_PUBLIC_API_BASE || '';
        // 모든 파일에 skip_reindex=true 추가
        const url = `${base}/api/rag/upload?skip_reindex=true`;

        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Authorization': `Basic ${window.btoa(`${auth.username}:${auth.password}`)}`,
          },
          body: formData,
        });

        const result = await response.json();

        if (result.status === 'SUCCESS') {
          results.success.push(file.name);
        } else {
          results.failed.push({ name: file.name, error: result.error || '알 수 없는 오류' });
        }
      } catch (e) {
        results.failed.push({ name: file.name, error: e.message });
      }
    }

    // 2단계: 업로드 성공한 파일이 있으면 인덱스 한 번만 재빌드
    if (results.success.length > 0) {
      addLog?.('RAG 인덱스 재빌드', `${results.success.length}개 파일 처리`);
      try {
        await apiCall({
          endpoint: '/api/rag/reload',
          method: 'POST',
          auth,
          data: { force: true },
        });
      } catch (e) {
        console.error('인덱스 재빌드 실패:', e);
      }
    }

    // 결과 알림
    let message = '';
    if (results.success.length > 0) {
      message += `✅ ${results.success.length}개 파일 업로드 성공\n`;
    }
    if (results.failed.length > 0) {
      message += `❌ ${results.failed.length}개 파일 업로드 실패\n`;
      results.failed.forEach(f => {
        message += `  - ${f.name}: ${f.error}\n`;
      });
    }
    alert(message);

    setSelectedFiles([]);

    // 파일 목록 및 상태 새로고침
    await loadFiles();
    await loadStatus();

    setUploading(false);
  }, [selectedFiles, auth, addLog, loadFiles, loadStatus, apiCall]);

  // 파일 삭제 (단일)
  const handleFileDelete = useCallback(async (filename) => {
    if (!confirm(`"${filename}" 파일을 삭제하시겠습니까?`)) return;
    if (!auth) return;

    addLog?.('RAG 문서 삭제', filename);

    try {
      const res = await apiCall({
        endpoint: '/api/rag/delete',
        method: 'POST',
        auth,
        data: { filename },
      });

      if (res?.status === 'SUCCESS') {
        alert('파일이 삭제되었습니다.');
        setSelectedForDelete(prev => {
          const next = new Set(prev);
          next.delete(filename);
          return next;
        });
        await loadFiles();
        await loadStatus();
      } else {
        alert(`삭제 실패: ${res.error || '알 수 없는 오류'}`);
      }
    } catch (e) {
      alert(`삭제 실패: ${e.message}`);
    }
  }, [apiCall, auth, addLog, loadFiles, loadStatus]);

  // 파일 다중 삭제
  const handleMultiDelete = useCallback(async () => {
    if (selectedForDelete.size === 0) return;
    if (!confirm(`${selectedForDelete.size}개 파일을 삭제하시겠습니까?`)) return;
    if (!auth) return;

    setDeleting(true);
    const totalFiles = selectedForDelete.size;
    const results = { success: [], failed: [] };

    // 1단계: 모든 파일 삭제 (skip_reindex=true로 재빌드 건너뛰기)
    for (const filename of selectedForDelete) {
      addLog?.('RAG 문서 삭제', `${filename} (${results.success.length + results.failed.length + 1}/${totalFiles})`);

      try {
        const res = await apiCall({
          endpoint: '/api/rag/delete',
          method: 'POST',
          auth,
          data: { filename, skip_reindex: true },
        });

        if (res?.status === 'SUCCESS') {
          results.success.push(filename);
        } else {
          results.failed.push({ name: filename, error: res.error || '알 수 없는 오류' });
        }
      } catch (e) {
        results.failed.push({ name: filename, error: e.message });
      }
    }

    // 2단계: 삭제 성공한 파일이 있으면 인덱스 한 번만 재빌드
    if (results.success.length > 0) {
      addLog?.('RAG 인덱스 재빌드', `${results.success.length}개 파일 삭제 후 재빌드`);
      try {
        await apiCall({
          endpoint: '/api/rag/reload',
          method: 'POST',
          auth,
          data: { force: true },
          timeoutMs: 300000, // 5분
        });
      } catch (e) {
        console.error('인덱스 재빌드 실패:', e);
      }
    }

    // 결과 알림
    let message = '';
    if (results.success.length > 0) {
      message += `✅ ${results.success.length}개 파일 삭제 성공\n`;
    }
    if (results.failed.length > 0) {
      message += `❌ ${results.failed.length}개 파일 삭제 실패\n`;
      results.failed.forEach(f => {
        message += `  - ${f.name}: ${f.error}\n`;
      });
    }
    alert(message);

    setSelectedForDelete(new Set());
    await loadFiles();
    await loadStatus();
    setDeleting(false);
  }, [selectedForDelete, apiCall, auth, addLog, loadFiles, loadStatus]);

  // 삭제용 파일 선택 토글
  const toggleFileForDelete = useCallback((filename) => {
    setSelectedForDelete(prev => {
      const next = new Set(prev);
      if (next.has(filename)) {
        next.delete(filename);
      } else {
        next.add(filename);
      }
      return next;
    });
  }, []);

  // 전체 선택/해제
  const toggleSelectAll = useCallback(() => {
    if (selectedForDelete.size === files.length) {
      setSelectedForDelete(new Set());
    } else {
      setSelectedForDelete(new Set(files.map(f => f.filename)));
    }
  }, [files, selectedForDelete]);

  // OCR 업로드
  const handleOcrUpload = useCallback(async () => {
    if (!selectedOcrFile || !auth) return;

    setOcrUploading(true);
    setOcrResult(null);
    addLog?.('OCR 이미지 업로드', selectedOcrFile.name);

    try {
      const formData = new FormData();
      formData.append('file', selectedOcrFile);
      formData.append('save_to_rag', 'true');

      const base = process.env.NEXT_PUBLIC_API_BASE || '';
      const url = `${base}/api/ocr/extract`;

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Basic ${window.btoa(`${auth.username}:${auth.password}`)}`,
        },
        body: formData,
      });

      const result = await response.json();

      if (result.status === 'SUCCESS') {
        setOcrResult(result);
        setSelectedOcrFile(null);
        await loadFiles();
        await loadStatus();
      } else {
        alert(`OCR 실패: ${result.error || '알 수 없는 오류'}`);
      }
    } catch (e) {
      alert(`OCR 실패: ${e.message}`);
    } finally {
      setOcrUploading(false);
    }
  }, [selectedOcrFile, auth, addLog, loadFiles, loadStatus]);

  // 인덱스 재빌드
  const handleReindex = useCallback(async () => {
    if (!confirm('RAG 인덱스를 재빌드하시겠습니까?')) return;
    if (!auth) return;

    addLog?.('RAG 인덱스 재빌드', '');
    setLoading(true);

    try {
      const res = await apiCall({
        endpoint: '/api/rag/reload',
        method: 'POST',
        auth,
        data: { force: true },
        timeoutMs: 300000, // 5분 (재빌드에 시간이 오래 걸릴 수 있음)
      });

      if (res?.status === 'SUCCESS') {
        alert('인덱스가 재빌드되었습니다.');
        await loadStatus();
      } else {
        alert(`재빌드 실패: ${res.error || '알 수 없는 오류'}`);
      }
    } catch (e) {
      alert(`재빌드 실패: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth, addLog, loadStatus]);

  // LightRAG 빌드
  const handleLightragBuild = useCallback(async () => {
    if (!confirm('LightRAG 지식 그래프를 빌드하시겠습니까?\n(LLM API 호출 비용이 발생합니다)')) return;
    if (!auth) return;

    addLog?.('LightRAG 빌드', '');
    setLightragBuilding(true);

    try {
      const res = await apiCall({
        endpoint: '/api/lightrag/build',
        method: 'POST',
        auth,
        data: { forceRebuild: false },
        timeoutMs: 600000, // 10분
      });

      if (res?.status === 'SUCCESS') {
        alert(res.message || 'LightRAG 빌드가 시작되었습니다.');
        // 빌드 완료까지 주기적으로 상태 확인
        setTimeout(() => loadStatus(), 10000);
      } else {
        alert(`LightRAG 빌드 실패: ${res.error || '알 수 없는 오류'}`);
      }
    } catch (e) {
      alert(`LightRAG 빌드 실패: ${e.message}`);
    } finally {
      setLightragBuilding(false);
    }
  }, [apiCall, auth, addLog, loadStatus]);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  const formatDate = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="space-y-4">
      <SectionHeader title="RAG 문서 관리" subtitle="PDF 및 문서 업로드/관리" />

      {/* RAG 상태 */}
      <div className="rounded-3xl border border-cookie-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-black text-cookie-brown">RAG 시스템 상태</h3>
          <button
            onClick={loadStatus}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-2xl border border-cookie-brown/10 bg-white px-3 py-2 text-xs font-black text-cookie-brown/80 hover:bg-cookie-beige disabled:opacity-50"
            type="button"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            새로고침
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="rounded-2xl border border-cookie-brown/10 bg-cookie-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              {status?.rag_ready ? (
                <CheckCircle size={18} className="text-green-600" />
              ) : (
                <XCircle size={18} className="text-red-600" />
              )}
              <span className="text-xs font-black text-cookie-brown/70">인덱스 상태</span>
            </div>
            <div className="text-lg font-black text-cookie-brown">
              {status?.rag_ready ? '준비됨' : '비활성'}
            </div>
          </div>

          <div className="rounded-2xl border border-cookie-brown/10 bg-cookie-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={18} className="text-blue-600" />
              <span className="text-xs font-black text-cookie-brown/70">문서 수</span>
            </div>
            <div className="text-lg font-black text-cookie-brown">
              {status?.files_count || 0}
            </div>
          </div>

          <div className="rounded-2xl border border-cookie-brown/10 bg-cookie-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={18} className="text-purple-600" />
              <span className="text-xs font-black text-cookie-brown/70">청크 수</span>
            </div>
            <div className="text-lg font-black text-cookie-brown">
              {status?.chunks_count || 0}
            </div>
          </div>

          <div className="rounded-2xl border border-cookie-brown/10 bg-cookie-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={18} className="text-amber-600" />
              <span className="text-xs font-black text-cookie-brown/70">임베딩 모델</span>
            </div>
            <div className="text-sm font-bold text-cookie-brown">
              {status?.embed_model || '-'}
            </div>
          </div>
        </div>

        {/* Advanced RAG Features */}
        <div className="mt-4 rounded-2xl border border-indigo-200 bg-indigo-50/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap size={16} className="text-indigo-600" />
            <span className="text-xs font-black text-indigo-800">RAG 기능 동작 여부 (인덱스 재빌드 시 활성화)</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {/* Hybrid Search (BM25 + Vector) */}
            <div className="flex items-center gap-2">
              <Search size={14} className={status?.bm25_ready ? 'text-green-600' : 'text-cookie-brown/50'} />
              <div>
                <div className="text-xs font-black text-cookie-brown/80">Hybrid Search</div>
                <div className="text-[10px] text-cookie-brown/60">
                  {status?.bm25_available ? (
                    status?.bm25_ready ? (
                      <span className="text-green-600">BM25 + Vector ✓</span>
                    ) : (
                      <span className="text-amber-600">BM25 대기중</span>
                    )
                  ) : (
                    <span className="text-cookie-brown/50">미설치</span>
                  )}
                </div>
              </div>
            </div>

            {/* Reranking - 비활성화 */}
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-cookie-brown/50" />
              <div>
                <div className="text-xs font-black text-cookie-brown/80">Reranking</div>
                <div className="text-[10px] text-cookie-brown/60">
                  <span className="text-cookie-brown/50">비활성</span>
                </div>
              </div>
            </div>

            {/* Simple Knowledge Graph - 비활성화 */}
            <div className="flex items-center gap-2">
              <GitBranch size={14} className="text-cookie-brown/50" />
              <div>
                <div className="text-xs font-black text-cookie-brown/80">Simple KG</div>
                <div className="text-[10px] text-cookie-brown/60">
                  <span className="text-cookie-brown/50">비활성</span>
                </div>
              </div>
            </div>

            {/* LightRAG */}
            <div className="flex items-center gap-2">
              <Sparkles size={14} className={lightragStatus?.ready ? 'text-green-600' : 'text-cookie-brown/50'} />
              <div>
                <div className="text-xs font-black text-cookie-brown/80">LightRAG</div>
                <div className="text-[10px] text-cookie-brown/60">
                  {lightragStatus?.available ? (
                    lightragStatus?.ready ? (
                      <span className="text-green-600">
                        {lightragStatus?.docs_count || 0}개 문서 ✓
                      </span>
                    ) : (
                      <span className="text-amber-600">빌드 필요</span>
                    )
                  ) : (
                    <span className="text-cookie-brown/50">미설치</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* LightRAG 빌드 버튼 */}
          {lightragStatus?.available && auth?.user_role === '관리자' && (
            <div className="mt-3">
              <button
                onClick={handleLightragBuild}
                disabled={lightragBuilding || lightragStatus?.ready}
                className={`w-full rounded-xl border px-3 py-2.5 text-xs font-black flex items-center justify-center gap-2 transition-colors disabled:cursor-not-allowed ${
                  lightragStatus?.ready
                    ? 'border-green-300 bg-green-100 text-green-700 disabled:opacity-100'
                    : 'border-indigo-300 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 disabled:opacity-50'
                }`}
                type="button"
              >
                {lightragBuilding ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    LightRAG 빌드 중...
                  </>
                ) : lightragStatus?.ready ? (
                  <>
                    <CheckCircle size={14} />
                    빌드 완료
                  </>
                ) : (
                  <>
                    <Sparkles size={14} />
                    LightRAG 빌드
                  </>
                )}
              </button>
              <div className="mt-2 text-[10px] text-cookie-brown/60 text-center">
                경량 지식 그래프 빌드 • 듀얼 레벨 검색 지원 (local/global/hybrid)
              </div>
            </div>
          )}

          {/* LightRAG 미설치 안내 */}
          {!lightragStatus?.available && auth?.user_role === '관리자' && (
            <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3">
              <div className="flex items-center gap-2 text-xs text-amber-800">
                <AlertCircle size={14} />
                <span className="font-bold">LightRAG 미설치</span>
              </div>
              <div className="mt-1 text-[10px] text-amber-700">
                <code className="bg-amber-100 px-1 py-0.5 rounded">pip install lightrag-hku</code> 설치 후 서버 재시작
              </div>
            </div>
          )}
        </div>

        {status?.error && (
          <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-3 text-xs font-semibold text-red-800">
            <strong>오류:</strong> {status.error}
          </div>
        )}

        {/* 인덱스 재빌드 버튼 - 비용 발생으로 비활성화 */}
        <div
          className="relative mt-4"
          onMouseEnter={() => setShowReindexTooltip(true)}
          onMouseLeave={() => setShowReindexTooltip(false)}
        >
          <button
            disabled={true}
            className="w-full rounded-2xl border border-cookie-brown/20 bg-cookie-beige px-4 py-2 text-sm font-black text-cookie-brown/50 cursor-not-allowed flex items-center justify-center gap-2"
            type="button"
          >
            <RefreshCw size={16} />
            인덱스 재빌드 (비활성화됨)
          </button>

          {/* 커스텀 툴팁 */}
          <div
            className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
              showReindexTooltip
                ? 'opacity-100 visible translate-y-0'
                : 'opacity-0 invisible translate-y-1'
            }`}
          >
            <div className="bg-cookie-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle size={14} className="text-amber-400" />
                <span className="font-bold text-amber-400">기능 비활성화됨</span>
              </div>
              <p className="text-cookie-beige leading-relaxed">
                인덱스 재빌드는 Contextual Retrieval을 위해 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                현재 프로덕션 환경에서는 비활성화되어 있습니다.
              </p>
              <div className="mt-2 pt-2 border-t border-cookie-brown/30 text-cookie-brown/50 text-[10px]">
                활성화가 필요하면 관리자에게 문의하세요
              </div>
            </div>
            {/* 툴팁 화살표 */}
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 bg-cookie-brown rotate-45" />
          </div>
        </div>
      </div>

      {/* RAG 모드 선택 */}
      <div className="rounded-3xl border border-cookie-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <Search size={18} className="text-indigo-600" />
          <h3 className="text-sm font-black text-cookie-brown">AI 에이전트 RAG 검색 모드</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          {/* 기본 RAG */}
          <button
            onClick={() => setSettings?.({ ...settings, ragMode: 'rag' })}
            className={`rounded-2xl border-2 p-4 text-left transition ${
              settings?.ragMode === 'rag'
                ? 'border-blue-500 bg-blue-50'
                : 'border-cookie-brown/10 bg-cookie-beige/30 hover:bg-cookie-beige/50'
            }`}
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Search size={16} className={settings?.ragMode === 'rag' ? 'text-blue-600' : 'text-cookie-brown/60'} />
              <span className={`text-sm font-black ${settings?.ragMode === 'rag' ? 'text-blue-700' : 'text-cookie-brown'}`}>
                RAG
              </span>
              {settings?.ragMode === 'rag' && (
                <CheckCircle size={14} className="text-blue-600 ml-auto" />
              )}
            </div>
            <div className="text-[11px] text-cookie-brown/70 leading-relaxed">
              FAISS + BM25
              <br />
              <span className="text-cookie-brown/50">싱글홉 질문에 최적</span>
            </div>
          </button>

          {/* LightRAG */}
          <button
            onClick={() => setSettings?.({ ...settings, ragMode: 'lightrag' })}
            className={`rounded-2xl border-2 p-4 text-left transition ${
              settings?.ragMode === 'lightrag'
                ? 'border-indigo-500 bg-indigo-50'
                : 'border-cookie-brown/10 bg-cookie-beige/30 hover:bg-cookie-beige/50'
            }`}
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className={settings?.ragMode === 'lightrag' ? 'text-indigo-600' : 'text-cookie-brown/60'} />
              <span className={`text-sm font-black ${settings?.ragMode === 'lightrag' ? 'text-indigo-700' : 'text-cookie-brown'}`}>
                LightRAG
              </span>
              {settings?.ragMode === 'lightrag' && (
                <CheckCircle size={14} className="text-indigo-600 ml-auto" />
              )}
            </div>
            <div className="text-[11px] text-cookie-brown/70 leading-relaxed">
              지식 그래프 기반
              <br />
              <span className="text-cookie-brown/50">멀티홉 질문에 최적 (+25%p)</span>
            </div>
          </button>

          {/* K²RAG (시험중 - 비활성화) */}
          <button
            disabled
            className="rounded-2xl border-2 p-4 text-left transition border-cookie-brown/10 bg-gray-100 opacity-50 cursor-not-allowed"
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm font-black text-gray-400">
                K²RAG
              </span>
              <span className="text-[9px] bg-yellow-200 text-yellow-700 px-1.5 py-0.5 rounded-full ml-auto font-bold">
                시험중
              </span>
            </div>
            <div className="text-[11px] text-gray-400 leading-relaxed">
              KG + Sub-Q + Hybrid
              <br />
              <span className="text-gray-300">고정밀 검색 (준비중)</span>
            </div>
          </button>

          {/* Auto (AI 판단) */}
          <button
            onClick={() => setSettings?.({ ...settings, ragMode: 'auto' })}
            className={`rounded-2xl border-2 p-4 text-left transition ${
              settings?.ragMode === 'auto' || !settings?.ragMode
                ? 'border-green-500 bg-green-50'
                : 'border-cookie-brown/10 bg-cookie-beige/30 hover:bg-cookie-beige/50'
            }`}
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap size={16} className={settings?.ragMode === 'auto' || !settings?.ragMode ? 'text-green-600' : 'text-cookie-brown/60'} />
              <span className={`text-sm font-black ${settings?.ragMode === 'auto' || !settings?.ragMode ? 'text-green-700' : 'text-cookie-brown'}`}>
                자동 선택
              </span>
              {(settings?.ragMode === 'auto' || !settings?.ragMode) && (
                <CheckCircle size={14} className="text-green-600 ml-auto" />
              )}
            </div>
            <div className="text-[11px] text-cookie-brown/70 leading-relaxed">
              AI가 질문에 맞게 선택
              <br />
              <span className="text-cookie-brown/50">두 방식 모두 사용 가능</span>
            </div>
          </button>
        </div>

        <div className="mt-3 text-[11px] text-cookie-brown/60 text-center">
          선택한 모드는 AI 에이전트가 세계관 질문에 답할 때 사용됩니다
        </div>
      </div>

      {/* 파일 업로드 - 비활성화됨 */}
      <div className="rounded-3xl border border-cookie-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur opacity-60">
        <h3 className="text-sm font-black text-cookie-brown mb-4">문서 업로드</h3>

        <div className="space-y-3">
          <div className="rounded-2xl border-2 border-dashed border-cookie-brown/20 bg-cookie-beige/50 p-6 text-center">
            <Upload size={32} className="mx-auto mb-3 text-cookie-brown/30" />
            <div className="inline-flex items-center gap-2 rounded-2xl border border-cookie-brown/20 bg-white/50 px-4 py-2 text-sm font-black text-cookie-brown/50 cursor-not-allowed">
              <Upload size={16} />
              파일 선택 (비활성화됨)
            </div>
          </div>

          <div
            className="relative"
            onMouseEnter={() => setShowUploadTooltip(true)}
            onMouseLeave={() => setShowUploadTooltip(false)}
          >
            <button
              disabled={true}
              className="w-full rounded-2xl border border-cookie-brown/20 bg-cookie-beige px-4 py-3 text-sm font-black text-cookie-brown/50 cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <Upload size={16} />
              문서 업로드 (비활성화됨)
            </button>

            {/* 커스텀 툴팁 */}
            <div
              className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
                showUploadTooltip
                  ? 'opacity-100 visible translate-y-0'
                  : 'opacity-0 invisible translate-y-1'
              }`}
            >
              <div className="bg-cookie-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle size={14} className="text-amber-400" />
                  <span className="font-bold text-amber-400">기능 비활성화됨</span>
                </div>
                <p className="text-cookie-beige leading-relaxed">
                  문서 업로드 시 Contextual Retrieval을 위해 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                  현재 프로덕션 환경에서는 비활성화되어 있습니다.
                </p>
                <div className="mt-2 pt-2 border-t border-cookie-brown/30 text-cookie-brown/50 text-[10px]">
                  활성화가 필요하면 관리자에게 문의하세요
                </div>
              </div>
              {/* 툴팁 화살표 */}
              <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 bg-cookie-brown rotate-45" />
            </div>
          </div>

          <div className="text-xs font-semibold text-cookie-brown/60">
            지원 형식: PDF, TXT, MD, JSON, CSV, LOG (파일당 최대 15MB)
          </div>
        </div>
      </div>

      {/* OCR 업로드 - 비활성화됨 */}
      <div className="rounded-3xl border border-cookie-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur opacity-60">
        <div className="flex items-center gap-2 mb-4">
          <ScanText size={18} className="text-purple-400" />
          <h3 className="text-sm font-black text-cookie-brown">OCR 이미지 업로드</h3>
        </div>

        <div className="space-y-3">
          <div className="rounded-2xl border-2 border-dashed border-purple-200 bg-purple-50/30 p-6 text-center">
            <Image size={32} className="mx-auto mb-3 text-purple-300" />
            <div className="inline-flex items-center gap-2 rounded-2xl border border-purple-200 bg-white/50 px-4 py-2 text-sm font-black text-purple-400 cursor-not-allowed">
              <Image size={16} />
              이미지 선택 (비활성화됨)
            </div>
          </div>

          <div
            className="relative"
            onMouseEnter={() => setShowOcrTooltip(true)}
            onMouseLeave={() => setShowOcrTooltip(false)}
          >
            <button
              disabled={true}
              className="w-full rounded-2xl border border-purple-300 bg-purple-100 px-4 py-3 text-sm font-black text-purple-400 cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <ScanText size={16} />
              OCR 추출 (비활성화됨)
            </button>

            {/* 커스텀 툴팁 */}
            <div
              className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
                showOcrTooltip
                  ? 'opacity-100 visible translate-y-0'
                  : 'opacity-0 invisible translate-y-1'
              }`}
            >
              <div className="bg-cookie-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle size={14} className="text-amber-400" />
                  <span className="font-bold text-amber-400">기능 비활성화됨</span>
                </div>
                <p className="text-cookie-beige leading-relaxed">
                  OCR 처리 후 RAG 저장 시 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                  현재 프로덕션 환경에서는 비활성화되어 있습니다.
                </p>
                <div className="mt-2 pt-2 border-t border-cookie-brown/30 text-cookie-brown/50 text-[10px]">
                  활성화가 필요하면 관리자에게 문의하세요
                </div>
              </div>
              {/* 툴팁 화살표 */}
              <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 bg-cookie-brown rotate-45" />
            </div>
          </div>

          <div className="text-xs font-semibold text-cookie-brown/60">
            지원 형식: JPG, PNG, BMP, TIFF, GIF, WEBP (최대 20MB) • 한국어/영어 지원
          </div>
        </div>
      </div>

      {/* 파일 목록 */}
      <div className="rounded-3xl border border-cookie-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-black text-cookie-brown">업로드된 문서</h3>
          <div className="flex items-center gap-2">
            {auth?.user_role === '관리자' && files.length > 0 && (
              <div
                className="relative"
                onMouseEnter={() => setShowDeleteTooltip(true)}
                onMouseLeave={() => setShowDeleteTooltip(false)}
              >
                <button
                  disabled={true}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-cookie-brown/10 bg-white/50 px-2.5 py-1.5 text-xs font-black text-cookie-brown/40 cursor-not-allowed"
                  type="button"
                >
                  <Trash2 size={14} />
                  삭제 (비활성화됨)
                </button>

                {/* 커스텀 툴팁 */}
                <div
                  className={`absolute bottom-full right-0 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
                    showDeleteTooltip
                      ? 'opacity-100 visible translate-y-0'
                      : 'opacity-0 invisible translate-y-1'
                  }`}
                >
                  <div className="bg-cookie-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle size={14} className="text-amber-400" />
                      <span className="font-bold text-amber-400">기능 비활성화됨</span>
                    </div>
                    <p className="text-cookie-beige leading-relaxed">
                      문서 삭제 후 인덱스 재빌드 시 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                      현재 프로덕션 환경에서는 비활성화되어 있습니다.
                    </p>
                    <div className="mt-2 pt-2 border-t border-cookie-brown/30 text-cookie-brown/50 text-[10px]">
                      활성화가 필요하면 관리자에게 문의하세요
                    </div>
                  </div>
                  {/* 툴팁 화살표 */}
                  <div className="absolute right-6 -bottom-1 w-2 h-2 bg-cookie-brown rotate-45" />
                </div>
              </div>
            )}
            <span className="text-xs font-black text-cookie-brown/60">{files.length}개</span>
          </div>
        </div>

        {files.length === 0 ? (
          <div className="rounded-2xl border border-cookie-brown/10 bg-cookie-beige/50 p-6 text-center text-sm font-semibold text-cookie-brown/60">
            업로드된 문서가 없습니다
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((file) => (
              <div
                key={file.filename}
                className="flex items-center justify-between gap-3 rounded-2xl border border-cookie-brown/10 bg-white p-3"
              >
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <FileText size={20} className="text-blue-600 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-bold text-cookie-brown truncate">{file.filename}</div>
                    <div className="text-xs font-semibold text-cookie-brown/60">
                      {formatBytes(file.size)} • {formatDate(file.modified)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
