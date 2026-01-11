import { useState, useCallback, useEffect } from 'react';
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

  const STORAGE_KEY_SESSION_ID = 'experiment_session_id';
  const STORAGE_KEY_ACCESS_CODE = 'experiment_access_code';
  const STORAGE_KEY_CONDITION = 'experiment_condition';

  const restoreSession = async (sid: string) => {
    try {
      const res = await fetch(`${API_BASE}/experiment/trial/${sid}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === "completed") {
          localStorage.removeItem(STORAGE_KEY_SESSION_ID);
          return;
        }
        // Restore state
        setSession({
          session_id: sid,
          condition: localStorage.getItem(STORAGE_KEY_CONDITION) || 'A'
        });
        setView('experiment');
        setCurrentTrial(data);
      }
    } catch (e) {
      console.error("Failed to restore session", e);
    }
  };

  useEffect(() => {
    const savedSid = localStorage.getItem(STORAGE_KEY_SESSION_ID);
    if (savedSid) {
      restoreSession(savedSid);
    }
  }, []);

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

      const finalCondition = data.condition || condition;

      setSession({ ...data, condition: finalCondition });

      // Save to storage
      localStorage.setItem(STORAGE_KEY_SESSION_ID, data.session_id);
      localStorage.setItem(STORAGE_KEY_CONDITION, finalCondition);

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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session.session_id, phase })
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
    // Try to get access code from formData or localStorage
    const codeToDelete = formData.access_code || localStorage.getItem(STORAGE_KEY_ACCESS_CODE);

    try {
      await fetch(`${API_BASE}/experiment/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          access_code: codeToDelete,
          ...formData,
          questionnaire: questData
        })
      });

      // Clear storage
      localStorage.removeItem(STORAGE_KEY_SESSION_ID);
      localStorage.removeItem(STORAGE_KEY_CONDITION);
      localStorage.removeItem(STORAGE_KEY_ACCESS_CODE);

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