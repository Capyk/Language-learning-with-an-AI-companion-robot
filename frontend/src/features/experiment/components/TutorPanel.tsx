import { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../../../config/api';

interface TutorMessage {
    id: string;
    sender: 'user' | 'tutor';
    text: string;
    isCorrection?: boolean;
    mnemonic?: string;
    example?: string;
}

interface TutorPanelProps {
    isVisible: boolean;
    currentTaskContext: {
        prompt: string;
        userAnswer?: string;
        expectedAnswer?: string;
        isCorrect?: boolean;
        exerciseType?: string;
    };
    nudge?: any;
    sessionId: string; // Add sessionId to props
    // Usunięto onClose, ponieważ nie jest używane
    language: 'de' | 'en';
    onLanguageChange: () => void;
}

// Usunięto onClose z destrukturyzacji propsów
export const TutorPanel = ({ isVisible, currentTaskContext, nudge, sessionId, language, onLanguageChange }: TutorPanelProps) => {

    const [messages, setMessages] = useState<TutorMessage[]>([
        {
            id: 'welcome',
            sender: 'tutor',
            text: language === 'de'
                ? "Hallo! Ich bin dein Tutor. Frag mich, wenn du Hilfe brauchst!"
                : "Hello! I am your tutor. Ask me if you need help!"
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        if (nudge) {
            const newMessage: TutorMessage = {
                id: `nudge-${Date.now()}`,
                sender: 'tutor',
                text: nudge.message || nudge.feedback || "Hier ist ein Hinweis.",
                isCorrection: !!nudge.correction,
                mnemonic: nudge.mnemonic,
                example: nudge.example
            };

            setMessages(prev => {
                const last = prev[prev.length - 1];
                if (last && last.text === newMessage.text) return prev;
                return [...prev, newMessage];
            });
        }
    }, [nudge]);

    const sendMessage = async (text: string) => {
        if (!text.trim()) return;

        const userMsg: TutorMessage = { id: `user-${Date.now()}`, sender: 'user', text };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE}/experiment/tutor/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId, // Send session_id
                    question: text,
                    task_context: {
                        prompt: currentTaskContext.prompt,
                        user_answer: currentTaskContext.userAnswer || null,
                        expected_answer: currentTaskContext.expectedAnswer || null,
                        is_correct: currentTaskContext.isCorrect,
                        exercise_type: currentTaskContext.exerciseType || 'image_labeling'
                    },
                    response_language: language
                })
            });

            const data = await response.json();
            const botMsg: TutorMessage = {
                id: `bot-${Date.now()}`,
                sender: 'tutor',
                text: data.message,
                isCorrection: !!data.correction,
                mnemonic: data.mnemonic,
                example: data.example
            };
            setMessages(prev => [...prev, botMsg]);

        } catch (e) {
            setMessages(prev => [...prev, { id: `err-${Date.now()}`, sender: 'tutor', text: language === 'de' ? "Entschuldigung, ich habe Verbindungsprobleme." : "Sorry, connection error." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const QuickChips = language === 'de'
        ? ["Einfacher erklären", "Gib ein Beispiel", "Was ist der Artikel?", "Merksatz bitte"]
        : ["Explain simply", "Give an example", "What is the article?", "Mnemonic please"];

    if (!isVisible) return null;

    return (
        <div className="flex flex-col h-[600px] w-full lg:w-[350px] bg-white rounded-[2rem] shadow-xl border-4 border-slate-100 overflow-hidden shrink-0 transition-all animate-in slide-in-from-right-4 duration-500">
            {/* Header */}
            <div className="bg-purple-600 p-4 text-white flex justify-between items-center shadow-md z-10">
                <div className="flex items-center gap-3">
                    <div className="bg-white p-1.5 rounded-full">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600"><path d="M12 2a8 8 0 0 0-8 8c0 5 3 9 8 9s8-4 8-9a8 8 0 0 0-8-8Z" /><path d="M10 11a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" /><path d="M15 16s-2 2-5-2" /></svg>
                    </div>
                    <span className="font-bold text-lg">{language === 'de' ? 'Dein Tutor' : 'Your Tutor'}</span>
                </div>

                <div className="flex items-center gap-2">
                    <span className="text-xs font-medium opacity-90">EN</span>
                    <button
                        onClick={onLanguageChange}
                        className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${language === 'de' ? 'bg-green-400' : 'bg-gray-300'}`}
                        aria-label="Toggle language"
                    >
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-md transition-transform duration-300 ${language === 'de' ? 'translate-x-6' : 'translate-x-0'}`} />
                    </button>
                    <span className="text-xs font-medium opacity-90">DE</span>
                </div>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 bg-slate-50 space-y-4 scroll-smooth">
                {messages.map(msg => (
                    <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
                        <div className={`max-w-[85%] p-3 rounded-2xl text-sm font-medium leading-relaxed shadow-sm ${msg.sender === 'user'
                            ? 'bg-purple-100 text-purple-900 rounded-tr-none'
                            : 'bg-white text-slate-700 border border-slate-100 rounded-tl-none'
                            }`}>
                            {msg.text}
                        </div>

                        {msg.sender === 'tutor' && (
                            <div className="space-y-1 mt-1 max-w-[85%] flex flex-wrap gap-1">
                                {msg.mnemonic && (
                                    <span className="inline-block px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded-lg font-bold border border-yellow-200">
                                        💡 {msg.mnemonic}
                                    </span>
                                )}
                                {msg.example && (
                                    <span className="inline-block px-2 py-1 bg-blue-50 text-blue-600 text-xs rounded-lg font-bold border border-blue-100 ml-1">
                                        📝 {msg.example}
                                    </span>
                                )}
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div className="flex items-center gap-2 text-slate-400 text-xs font-bold pl-2">
                        <span className="animate-bounce">●</span>
                        <span className="animate-bounce delay-100">●</span>
                        <span className="animate-bounce delay-200">●</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white border-t border-slate-100">
                <div className="flex gap-2 overflow-x-auto pb-2 mb-2 no-scrollbar">
                    {QuickChips.map(chip => (
                        <button
                            key={chip}
                            onClick={() => sendMessage(chip)}
                            disabled={isLoading}
                            className="whitespace-nowrap px-3 py-1 bg-slate-100 hover:bg-purple-50 hover:text-purple-600 hover:border-purple-200 border border-transparent rounded-full text-xs font-bold text-slate-500 transition-all active:scale-95 shrink-0"
                        >
                            {chip}
                        </button>
                    ))}
                </div>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && sendMessage(input)}
                        placeholder={language === 'de' ? "Frag mich etwas..." : "Ask me something..."}
                        className="flex-1 bg-slate-50 border-slate-200 border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-purple-400 focus:bg-white transition-all text-slate-800"
                        disabled={isLoading}
                    />
                    <button
                        onClick={() => sendMessage(input)}
                        disabled={isLoading || !input.trim()}
                        className="p-2 bg-purple-600 text-white rounded-xl hover:bg-purple-700 active:scale-95 transition-all disabled:opacity-50 disabled:scale-100 flex items-center justify-center"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>
                    </button>
                </div>
            </div>
        </div>
    );
};