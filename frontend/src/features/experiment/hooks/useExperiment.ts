import { useState, useCallback } from 'react';
import { API_BASE } from '../../../config/api';

export const useExperiment = () => {
  const [session, setSession] = useState<any>(null);
  const [currentTrial, setCurrentTrial] = useState<any>(null);
  const [feedback, setFeedback] = useState<any>(null);
  const [questData, setQuestData] = useState<any>(null);
  const [nudge, setNudge] = useState<any>(null);
  
  // NOWE: Globalny stan języka
  const [language, setLanguage] = useState<'de' | 'en'>('de');

  const [view, setView] = useState<'intro' | 'experiment' | 'questionnaire' | 'demographics' | 'done'>('intro');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchNextTrial = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setFeedback(null);
    setNudge(null);
    
    try {
      const resp = await fetch(`${API_BASE}/experiment/trial/${sessionId}`);
      const data = await resp.json();
      
      console.log("Fetched trial data:", data);

      if (data.status === "completed") {
        setView('questionnaire');
      } else if (data.status === "transition") {
        console.log("Transitioning phases...");
        setTimeout(() => {
            fetchNextTrial(sessionId);
        }, 500); 
      } else {
        setCurrentTrial(data);
      }
    } catch (err) { 
      console.error(err);
      setError("Failed to load task."); 
    } finally { 
      setIsLoading(false); 
    }
  }, []);

  const startExperiment = async (condition: 'A' | 'B') => {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/experiment/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: `user_${Date.now()}`, condition }),
      });
      const data = await resp.json();
      setSession(data);
      setView('experiment');
      fetchNextTrial(data.session_id);
    } catch { 
      setError("Connection failed. Check backend."); 
      setIsLoading(false);
    }
  };

  const submitAnswer = async (userAnswer: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const resp = await fetch(`${API_BASE}/experiment/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            session_id: session.session_id, 
            user_answer: userAnswer, 
            start_time: 0 
        }),
      });
      const data = await resp.json();
      
      console.log("Submit response:", data);

      const shouldMoveNext = 
        data.move_next || 
        data.status === "transition" ||
        ((currentTrial?.phase === 'pre-test' || currentTrial?.phase === 'post-test') && !data.feedback);

      if (shouldMoveNext) {
        fetchNextTrial(session.session_id);
      } else {
        if (data.feedback || data.score !== undefined) {
            setFeedback(data);
        }
        if (data.nudge) {
            setNudge(data.nudge);
        }
      }
    } catch (err) {
      console.error(err);
      setError("Submission error.");
      setIsLoading(false);
    }
  };

  const skipToPhase = async (phase: string) => {
    if (!session) return;
    setIsLoading(true);
    try {
        await fetch(`${API_BASE}/experiment/skip`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: session.session_id, phase})
        });
        fetchNextTrial(session.session_id);
    } catch { 
        setError("Skip failed"); 
        setIsLoading(false);
    }
  };

  const handleQuestSubmit = (data: any) => {
    setQuestData(data);
    setView('demographics');
  };

  const handleFinalSubmit = async (formData: any) => {
    setIsLoading(true);
    try {
        await fetch(`${API_BASE}/experiment/finalize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: session.session_id,
                ...formData,
                questionnaire: questData
            })
        });
        setView('done');
    } catch {
        setError("Failed to save data.");
    } finally {
        setIsLoading(false);
    }
  };

  return {
    state: { session, currentTrial, feedback, nudge, isLoading, error, view, language }, // Dodano language
    actions: { 
        startExperiment, 
        submitAnswer, 
        fetchNextTrial, 
        skipToPhase, 
        handleQuestSubmit, 
        handleFinalSubmit,
        setLanguage // Dodano setLanguage
    }
  };
};