import { useState } from 'react';
import { Icon } from '../../../components/ui/Icons';
import { API_BASE } from '../../../config/api';

interface LandingPageProps {
    onSuccess: (code: string, group: string) => void;
}

export const LandingPage = ({ onSuccess }: LandingPageProps) => {
    const [code, setCode] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        if (!code.trim()) {
            setError("Please enter a code.");
            setLoading(false);
            return;
        }

        try {
            const res = await fetch(`${API_BASE}/experiment/auth`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ access_code: code.trim() })
            });
            const data = await res.json();
            if (res.ok) {
                // Save code for persistence
                localStorage.setItem('experiment_access_code', code.trim());
                onSuccess(data.token, data.group);
            } else {
                setError(data.detail || "Invalid access code.");
            }
        } catch (err) {
            setError("Connection error. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-md bg-white rounded-[2.5rem] shadow-2xl p-10 border-8 border-white mx-auto text-center animate-in fade-in zoom-in duration-500">
            <div className="bg-indigo-50 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 text-indigo-600">
                <Icon.Lock size={40} />
            </div>
            <h1 className="text-3xl font-black text-slate-900 mb-2">Welcome</h1>
            <p className="text-slate-500 mb-8">Please enter your access code to begin.</p>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                    <input
                        type="text"
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        placeholder="XXX-XXX"
                        className="w-full py-4 px-6 rounded-2xl bg-slate-50 border-2 border-slate-200 focus:border-indigo-500 focus:bg-white transition-all text-center text-xl font-bold tracking-widest uppercase text-slate-800 outline-none"
                    />
                </div>

                {error && (
                    <div className="text-red-500 text-sm font-bold bg-red-50 p-3 rounded-xl flex items-center justify-center gap-2 animate-bounce">
                        <Icon.Info size={16} />
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-4 bg-indigo-600 text-white rounded-2xl font-bold text-lg hover:bg-indigo-700 shadow-xl transition-all active:scale-95 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                    {loading ? "Verifying..." : "Enter Experiment"}
                </button>
            </form>

        </div>
    );
};
