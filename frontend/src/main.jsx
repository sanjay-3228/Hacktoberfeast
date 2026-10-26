import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowDownUp, ArrowRight, AudioLines, BookOpen, Check, ChevronDown,
  Clock3, CloudUpload, FileText, FolderOpen, Grid2X2, LayoutDashboard, List, Play, Plus,
  Search, Sparkles, Video, X, MessageCircle, Send,
  UploadCloud, FileVideo, CheckCircle2, Mic2
} from 'lucide-react';
import './styles.css';

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const notebookColors = ['lavender', 'blue', 'orange', 'green', 'pink', 'teal'];

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remaining).padStart(2, '0')}` : `${minutes}:${String(remaining).padStart(2, '0')}`;
}

function resultTranscript(result) {
  const names = Object.fromEntries((result.speakers || []).map(speaker => [speaker.id, speaker.name || speaker.id]));
  return (result.messages || []).map(message => `${names[message.speaker] || message.speaker} · ${formatDuration(message.start)}\n${message.text}`).join('\n\n');
}

function backendNotebook(summary, result = null) {
  const data = result || summary;
  return {
    id: summary.job_id,
    backendId: summary.job_id,
    title: data.title || summary.title || summary.job_id,
    description: data.topic || summary.topic || 'Processed media from the Video Knowledge Extractor pipeline.',
    type: 'Video',
    duration: formatDuration(data.duration || summary.duration),
    date: result ? 'Ready' : 'Available',
    tags: result?.speakers?.length ? [`${result.speakers.length} speakers`, 'Processed'] : ['Processed media'],
    color: notebookColors[Math.abs(String(summary.job_id).length) % notebookColors.length],
    progress: result ? 'Ready' : 'Loading',
    transcript: result ? resultTranscript(result) : 'Open this notebook to load its transcript and source-linked result.',
    result,
  };
}

async function apiFetch(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function App() {
  const [notebooks, setNotebooks] = useState([]);
  const [activeTab, setActiveTab] = useState('All notebooks');
  const [search, setSearch] = useState('');
  const [view, setView] = useState('grid');
  const [sort, setSort] = useState('Recently updated');
  const [showSort, setShowSort] = useState(false);
  const [selected, setSelected] = useState(null);
  const [showCreate, setShowCreate] = useState(false);
  const [files, setFiles] = useState([]);
  const [newTitle, setNewTitle] = useState('');
  const [expectedSpeakers, setExpectedSpeakers] = useState('');
  const [notice, setNotice] = useState('');
  const [backendState, setBackendState] = useState('connecting');
  const [question, setQuestion] = useState('');
  const [qaHistory, setQaHistory] = useState([]);
  const [asking, setAsking] = useState(false);
  const fileInput = useRef(null);
  const dropRef = useRef(null);
  const videoRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch('/jobs')
      .then(jobs => {
        if (cancelled) return;
        setNotebooks(jobs.map(job => backendNotebook(job)));
        setBackendState('connected');
      })
      .catch(() => {
        if (!cancelled) {
          setNotebooks([]);
          setBackendState('offline');
          setNotice('Backend unavailable. Start the API to load notebooks.');
        }
      });
    return () => { cancelled = true; };
  }, []);

  const filtered = useMemo(() => {
    let list = [...notebooks];
    if (activeTab === 'Videos') list = list.filter(n => n.type === 'Video');
    if (activeTab === 'Audio') list = list.filter(n => n.type === 'Audio');
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(n => `${n.title} ${n.description} ${n.tags.join(' ')}`.toLowerCase().includes(q));
    }
    if (sort === 'Title A–Z') list.sort((a,b) => a.title.localeCompare(b.title));
    if (sort === 'Duration') list.sort((a,b) => parseDuration(a.duration) - parseDuration(b.duration));
    if (sort === 'Recently updated') list.sort((a,b) => a.id < b.id ? 1 : -1);
    return list;
  }, [notebooks, activeTab, search, sort]);

  function parseDuration(s) { const p = s.split(':').map(Number); return p.length === 3 ? p[0]*3600+p[1]*60+p[2] : p[0]*60+p[1]; }
  function chooseFiles(fileList) { const picked = Array.from(fileList || []); if (!picked.length) return; setFiles(prev => [...prev, ...picked]); setNotice(`${picked.length} file${picked.length > 1 ? 's' : ''} added. Create a notebook to continue.`); }
  async function openNotebook(notebook) {
    setQaHistory([]);
    setQuestion('');
    try {
      const result = await apiFetch(`/jobs/${encodeURIComponent(notebook.backendId)}/result`);
      setSelected({ ...notebook, ...backendNotebook({ job_id: notebook.backendId, title: notebook.title, topic: notebook.description, duration: parseDuration(notebook.duration) }, result) });
    } catch (error) {
      setNotice(error.message);
      setSelected(notebook);
    }
  }

  async function askAboutConversation(e) {
    e.preventDefault();
    const prompt = question.trim();
    if (!selected || !prompt || asking) return;
    setAsking(true);
    try {
      const response = await apiFetch(`/jobs/${encodeURIComponent(selected.backendId)}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: prompt }),
      });
      setQaHistory(previous => [...previous, { question: prompt, ...response }]);
      setQuestion('');
    } catch (error) {
      setNotice(`Could not answer question: ${error.message}`);
    } finally {
      setAsking(false);
    }
  }

  function jumpToSource(seconds) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Number(seconds) || 0;
    videoRef.current.play().catch(() => {});
  }

  async function waitForJob(jobId) {
    for (let attempt = 0; attempt < 600; attempt += 1) {
      const status = await apiFetch(`/jobs/${encodeURIComponent(jobId)}/status`);
      if (status.status === 'done') return apiFetch(`/jobs/${encodeURIComponent(jobId)}/result`);
      if (status.status === 'error') throw new Error(status.detail || 'Pipeline processing failed');
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
    throw new Error('Pipeline processing timed out');
  }

  async function handleCreate(e) {
    e?.preventDefault();
    const title = newTitle.trim() || files[0]?.name.replace(/\.[^.]+$/, '') || 'Untitled recording';
    if (files.length) {
      const form = new FormData();
      form.append('file', files[0]);
      form.append('title', title);
      form.append('use_faces', 'false');
      form.append('use_llm', 'true');
      form.append('embed', 'false');
      if (expectedSpeakers.trim()) form.append('num_speakers', expectedSpeakers.trim());
      try {
        const queued = await apiFetch('/jobs', { method: 'POST', body: form });
        const queuedNotebook = backendNotebook({ job_id: queued.job_id, title, topic: 'Processing uploaded media.', duration: 0 });
        setNotebooks(prev => [queuedNotebook, ...prev.filter(notebook => notebook.backendId !== queued.job_id)]);
        setSelected(queuedNotebook);
        setShowCreate(false); setFiles([]); setNewTitle(''); setExpectedSpeakers('');
        setNotice('Recording uploaded. Pipeline processing has started.');
        const result = await waitForJob(queued.job_id);
        const readyNotebook = backendNotebook({ job_id: queued.job_id, title, topic: result.topic, duration: result.duration }, result);
        setNotebooks(prev => prev.map(notebook => notebook.backendId === queued.job_id ? readyNotebook : notebook));
        setSelected(readyNotebook);
        setNotice('Notebook processing complete.');
      } catch (error) {
        setNotice(`Could not process recording: ${error.message}`);
      }
      return;
    }
  }
  function handleDrop(e) { e.preventDefault(); dropRef.current?.classList.remove('dragging'); chooseFiles(e.dataTransfer.files); }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><AudioLines size={19}/></div><span>synchra</span><span className="brand-beta">BETA</span></div>
      <button className="workspace-select"><div className="workspace-icon">V</div><span><strong>VKE workspace</strong><small>Backend jobs</small></span><ChevronDown size={15}/></button>
      <div className="side-section-label">WORKSPACE</div>
      <nav className="side-nav">
        <button className="nav-item active"><LayoutDashboard size={17}/><span>Overview</span></button>
        <button className="nav-item" onClick={() => {setActiveTab('All notebooks'); document.getElementById('library')?.scrollIntoView({behavior:'smooth'});}}><BookOpen size={17}/><span>Notebooks</span><span className="nav-count">{notebooks.length}</span></button>
      </nav>
      <div className="side-section-label library-label">LIBRARY <button onClick={() => setShowCreate(true)} title="New notebook"><Plus size={14}/></button></div>
      <nav className="side-nav library-nav">
        <button className={`nav-item ${activeTab==='All notebooks'?'sub-active':''}`} onClick={() => setActiveTab('All notebooks')}><FolderOpen size={16}/><span>All notebooks</span></button>
        <button className={`nav-item ${activeTab==='Videos'?'sub-active':''}`} onClick={() => setActiveTab('Videos')}><Video size={16}/><span>Video recordings</span></button>
      </nav>
      <div className="sidebar-bottom" />
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Overview</strong></div><div className={`connection-status ${backendState}`}>{backendState === 'connected' ? 'Backend connected' : backendState === 'offline' ? 'Backend unavailable' : 'Connecting to backend'}</div></header>
      <div className="content-wrap">
        <section className="welcome-row">
          <div><div className="eyebrow"><span className="eyebrow-dot"/> AI RESEARCH NOTEBOOK</div><h1>Welcome to Synchra</h1><p className="welcome-copy">Your personal AI research notebook for video recordings, transcripts, and grounded<br className="desktop-break"/> synthesis.</p></div>
          <div className="summary-pill"><span><FolderOpen size={14}/> {notebooks.length} Notebooks</span></div>
        </section>

        <section className="hero-actions">
          <div className="create-card"><div className="card-top"><div className="create-icon"><Plus size={21}/></div><span className="micro-badge">Backend processing</span></div><div className="create-copy"><h2>New notebook</h2><p>Upload a recording and the backend pipeline will create its transcript, searchable chunks, speakers, and AI analysis.</p></div><button className="primary-button" onClick={() => setShowCreate(true)}>Process recording <ArrowRight size={15}/></button><div className="card-glow"/></div>
          <div className={`upload-zone ${files.length ? 'has-files' : ''}`} ref={dropRef} onDragOver={e => {e.preventDefault();dropRef.current?.classList.add('dragging')}} onDragLeave={() => dropRef.current?.classList.remove('dragging')} onDrop={handleDrop}>
            <input ref={fileInput} type="file" accept="video/*,audio/*,.mkv,.mov,.mp4,.mp3,.wav,.m4a" multiple hidden onChange={e => chooseFiles(e.target.files)}/>
            <div className="upload-icon"><CloudUpload size={23}/></div>
            {files.length ? <><h3>{files.length} file{files.length>1?'s':''} ready to add</h3><p className="file-list">{files.slice(0,2).map(f => f.name).join(' · ')}{files.length > 2 ? ` +${files.length-2} more` : ''}</p><div className="upload-buttons"><button className="outline-button" onClick={() => fileInput.current?.click()}>Add more</button><button className="text-button" onClick={() => setFiles([])}>Clear</button></div></> : <><h3>Drop recording files to generate a notebook, or <button className="inline-link" onClick={() => fileInput.current?.click()}>browse</button></h3><p>Supports MP4, MOV, MKV, MP3 up to 2GB. Speaker diarization, grounded<br className="wide-break"/> citations, and transcriptions are automated.</p></>}
          </div>
        </section>

        <section className="library-section" id="library">
          <div className="library-header"><div className="filter-tabs">{['All notebooks','Videos'].map(tab => <button key={tab} className={activeTab===tab?'selected':''} onClick={() => setActiveTab(tab)}>{tab}{tab==='All notebooks' && <span className="tab-count">{notebooks.length}</span>}</button>)}</div><div className="library-tools"><div className="search-field"><Search size={15}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search notebooks..." aria-label="Search notebooks"/>{search && <button onClick={() => setSearch('')}><X size={13}/></button>}</div><div className="sort-wrap"><button className="tool-button" onClick={() => setShowSort(v=>!v)}><ArrowDownUp size={14}/><span>Sort</span><ChevronDown size={13}/></button>{showSort && <div className="sort-menu">{['Recently updated','Title A–Z','Duration'].map(s => <button key={s} onClick={() => {setSort(s);setShowSort(false)}}>{s}{sort===s&&<Check size={13}/>}</button>)}</div>}</div><div className="view-toggle"><button className={view==='grid'?'chosen':''} onClick={() => setView('grid')} title="Grid view"><Grid2X2 size={15}/></button><button className={view==='list'?'chosen':''} onClick={() => setView('list')} title="List view"><List size={16}/></button></div></div></div>

          <div className={view==='grid'?'notebook-grid':'notebook-list'}>
            {filtered.map(n => <article className={`notebook-card ${view==='list'?'list-card':''}`} key={n.id}>
              <div className="notebook-main"><div className="notebook-meta"><span className="media-type"><FileVideo size={12}/> {n.type}</span><span className="dot-separator">·</span><span>{n.date}</span></div><button className="notebook-title" onClick={() => openNotebook(n)}>{n.title}</button><p className="notebook-description">{n.description}</p><div className={`media-preview ${n.color}`}><div className="preview-pattern"><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/></div><div className="preview-bottom"><div className="source-stack"><span><Video size={13}/></span><span><FileText size={12}/></span><span><Sparkles size={12}/></span><small>Processed video</small></div><span className="duration-pill"><Play size={10} fill="currentColor"/> {n.duration}</span></div></div><div className="tag-row">{n.tags.map(t => <span key={t} className="topic-tag">{t}</span>)}</div></div>
              <footer className="notebook-footer"><div className="footer-stats"><span><FileText size={13}/> Transcript</span><span><Sparkles size={13}/> {n.progress==='Ready'?'AI notes':'Draft'}</span></div><button className="open-link" onClick={() => openNotebook(n)}>Open notebook <ArrowRight size={13}/></button></footer>
            </article>)}
            {filtered.length===0 && <div className="empty-state"><div className="empty-icon"><Search size={20}/></div><h3>No notebooks found</h3><p>Try a different search or choose another library filter.</p><button className="outline-button" onClick={() => {setSearch('');setActiveTab('All notebooks')}}>Clear filters</button></div>}
          </div>
          <div className="library-bottom"><span>Showing <strong>{filtered.length}</strong> of <strong>{notebooks.length}</strong> notebooks</span><button onClick={() => setShowCreate(true)}><Plus size={14}/> New notebook</button></div>
        </section>

        <footer className="page-footer"><span>© 2026 Synchra</span><span>Connected to the Video Knowledge Extractor backend</span></footer>
      </div>
    </main>

    {notice && <div className="toast" role="status"><CheckCircle2 size={17}/><span>{notice}</span><button onClick={() => setNotice('')}><X size={14}/></button></div>}

    {showCreate && <div className="modal-backdrop" onMouseDown={e => {if(e.target===e.currentTarget)setShowCreate(false)}}><form className="create-modal" onSubmit={handleCreate}><div className="modal-header"><div className="modal-heading-icon"><Plus size={18}/></div><div><h2>Process a recording</h2><p>The backend will create a searchable notebook.</p></div><button type="button" className="close-modal" onClick={() => setShowCreate(false)}><X size={18}/></button></div><label>Notebook name <span>(optional)</span><input value={newTitle} onChange={e => setNewTitle(e.target.value)} placeholder="e.g. Product discovery interviews" autoFocus/></label><label>Expected speakers <span>(optional; leave blank to estimate)</span><input type="number" min="1" max="32" value={expectedSpeakers} onChange={e => setExpectedSpeakers(e.target.value)} placeholder="e.g. 2"/></label><div className="modal-file-row"><div className="mini-upload"><UploadCloud size={17}/><span>{files.length ? `${files.length} file${files.length>1?'s':''} selected` : 'Add local recording'}</span></div><button type="button" className="outline-button" onClick={() => fileInput.current?.click()}>Browse files</button></div>{files.length>0&&<div className="modal-selected-files">{files.map((f,i)=><span key={i}><FileText size={12}/>{f.name}<button type="button" onClick={()=>setFiles(prev=>prev.filter((_,j)=>j!==i))}><X size={12}/></button></span>)}</div>}<div className="modal-actions"><button type="button" className="text-button" onClick={() => setShowCreate(false)}>Cancel</button><button type="submit" className="primary-button" disabled={!files.length}>Start processing <ArrowRight size={14}/></button></div></form></div>}

    {selected && <div className="modal-backdrop" onMouseDown={e => {if(e.target===e.currentTarget)setSelected(null)}}><div className="detail-modal"><div className="detail-top"><span className="media-type"><FileVideo size={12}/> Video notebook</span><div className="detail-controls"><button className="close-modal" onClick={() => setSelected(null)}><X size={18}/></button></div></div><h2>{selected.title}</h2><p className="detail-description">{selected.description}</p><video ref={videoRef} className="backend-video" controls preload="metadata" src={`${API_BASE_URL}/jobs/${encodeURIComponent(selected.backendId)}/video`} />{selected.result?.topic && <div className="backend-topic"><strong>Topic</strong><p>{selected.result.topic}</p></div>}<div className="detail-tabs"><span className="detail-tab-active"><FileText size={14}/> Transcript</span><span><Clock3 size={14}/> {selected.duration}</span></div><div className="summary-block"><div className="summary-heading"><Sparkles size={15}/> Processed result <span className="grounded-label">BACKEND RESULT</span></div><p>{selected.transcript}</p></div>{selected.result?.takeaways?.length > 0 && <div className="attached-files"><strong>Key takeaways</strong>{selected.result.takeaways.map(takeaway => <span key={takeaway}><Sparkles size={13}/>{takeaway}</span>)}</div>}<section className="qa-panel"><div className="qa-heading"><div><MessageCircle size={15}/><strong>Ask about this conversation</strong></div><span>Answers cite transcript moments</span></div>{qaHistory.length > 0 && <div className="qa-history" role="log" aria-live="polite">{qaHistory.map((turn, index) => <article className="qa-turn" key={`${index}-${turn.question}`}><p className="qa-user-question">{turn.question}</p><p className="qa-answer">{turn.answer}</p>{turn.sources?.length > 0 && <div className="qa-sources"><span>Sources</span>{turn.sources.map((source, sourceIndex) => <button type="button" className="qa-source" key={`${source.start}-${sourceIndex}`} onClick={() => jumpToSource(source.start)}>[{sourceIndex + 1}] {formatDuration(source.start)} · {source.speaker}</button>)}</div>}</article>)}</div>}<form className="qa-form" onSubmit={askAboutConversation}><input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about what was said..." aria-label="Ask a question about this conversation" maxLength={4000}/><button className="qa-submit" type="submit" disabled={!question.trim() || asking} aria-label="Ask question">{asking ? <span className="qa-spinner"/> : <Send size={15}/>}</button></form></section><div className="detail-footer"><span><CheckCircle2 size={14}/> Loaded from backend</span><button className="primary-button" onClick={() => window.open(`${API_BASE_URL}/jobs/${encodeURIComponent(selected.backendId)}/report`, '_blank', 'noopener,noreferrer')}>Export full report <ArrowRight size={13}/></button></div></div></div>}
  </div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
