import { useState, useCallback } from 'react';
import { API_BASE } from '../../../config/api';

export const useExperiment = () => {
  const [session, setSession] = useState<any>(null);
  const [currentTrial, setCurrentTrial] = useState<any>(null);
  const [feedback, setFeedback] = useState<any>(null);
  const [questData, setQuestData] = useState<any>(null);
  
  const [view, setView] = useState<'intro' | 'experiment' | 'questionnaire' | 'demographics' | 'done'>('intro');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pobieranie kolejnego zadania
  const fetchNextTrial = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setFeedback(null); // WAŻNE: Reset feedbacku przy pobieraniu nowego zadania
    
    try {
      const resp = await fetch(`${API_BASE}/experiment/trial/${sessionId}`);
      const data = await resp.json();
      
      console.log("Fetched trial data:", data); // Debugowanie w konsoli

      if (data.status === "completed") {
        setView('questionnaire');
      } else if (data.status === "transition") {
        // Jeśli backend mówi "transition", czekamy chwilę i pytamy ponownie
        // (Daje to czas backendowi na przetworzenie zmiany fazy)
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

  // Rozpoczęcie eksperymentu
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

  // Wysyłanie odpowiedzi
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
      
      console.log("Submit response:", data); // Debugowanie

      // LOGIKA PRZEJŚCIA:
      // 1. Backend każe iść dalej (move_next: true)
      // 2. LUB Backend zwraca status transition
      // 3. LUB Jesteśmy w fazie testowej (pre/post) i nie ma feedbacku (zabezpieczenie przed utknięciem)
      const shouldMoveNext = 
        data.move_next || 
        data.status === "transition" ||
        ((currentTrial?.phase === 'pre-test' || currentTrial?.phase === 'post-test') && !data.feedback);

      if (shouldMoveNext) {
        fetchNextTrial(session.session_id);
      } else {
        // Pokaż feedback tylko jeśli backend go zwrócił i nie każe iść dalej
        setFeedback(data);
      }
    } catch (err) {
      console.error(err);
      setError("Submission error.");
      setIsLoading(false);
    }
  };

  // Przeskakiwanie faz (dla dewelopera/testów)
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

  // Wewnątrz useExperiment.ts

  // ZASTĄP STARE handleQuestSubmit i handleFinalSubmit TYM KODEM:

  const handleQuestSubmit = (data: any) => {
    setQuestData(data);
    setView('demographics');
  };

// Krok 2: Wyślij wszystko do backendu
  const handleFinalSubmit = async (formData: any) => {
    setIsLoading(true);
    try {
        await fetch(`${API_BASE}/experiment/finalize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: session.session_id,
                ...formData,           // Dane demograficzne (age, gender...)
                questionnaire: questData // Dane z poprzedniego kroku
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
    state: { session, currentTrial, feedback, isLoading, error, view },
    actions: { 
        startExperiment, 
        submitAnswer, 
        fetchNextTrial, 
        skipToPhase, 
        handleQuestSubmit, 
        handleFinalSubmit 
    }
  };
};