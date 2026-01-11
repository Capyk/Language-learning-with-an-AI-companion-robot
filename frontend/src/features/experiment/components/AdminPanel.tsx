import { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';
import { API_BASE } from '../../../config/api';

interface AccessCode {
    code: string;
    used: boolean;
    created_at: string;
    copy_count: number;
}

export const AdminPanel = () => {
    const [codes, setCodes] = useState<AccessCode[]>([]);
    const [loading, setLoading] = useState(false);
    const [genCount, setGenCount] = useState(10);
    const [token, setToken] = useState("");
    const [isLoggedIn, setIsLoggedIn] = useState(false);

    const [passwordInput, setPasswordInput] = useState("");

    const fetchCodes = async (authToken: string) => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/admin/codes`, {
                headers: { 'X-Admin-Token': authToken }
            });
            if (res.status === 403) {
                setIsLoggedIn(false);
                return;
            }
            const data = await res.json();
            if (data.codes) setCodes(data.codes);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isLoggedIn) fetchCodes(token);
    }, [isLoggedIn]);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setToken(passwordInput);
        setIsLoggedIn(true);
        // We could verify here, but let's just let fetchCodes fail if wrong
    };

    const generateCodes = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/admin/codes/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Admin-Token': token
                },
                body: JSON.stringify({ count: genCount })
            });
            await fetchCodes(token);
        } catch (e) {
            alert("Error generating codes");
        } finally {
            setLoading(false);
        }
    };

    const handleCopy = async (code: string) => {
        try {
            await navigator.clipboard.writeText(code);
            // Notify backend to increment counter
            await fetch(`${API_BASE}/admin/codes/${code}/copy`, {
                method: 'POST',
                headers: { 'X-Admin-Token': token }
            });
            // Refresh list to show count
            fetchCodes(token); // Update counts
        } catch (e) {
            alert("Copy failed");
        }
    };

    if (!isLoggedIn) {
        return (
            <div className="fixed inset-0 w-screen h-screen bg-slate-50 overflow-auto grid place-items-center">
                <div className="bg-white p-10 rounded-3xl shadow-2xl border-4 border-white w-full max-w-md animate-in fade-in zoom-in">
                    <h1 className="text-2xl font-black text-slate-900 mb-6 text-center flex justify-center items-center gap-2">
                        <Icon.Lock size={28} className="text-slate-400" />
                        Admin Access
                    </h1>
                    <form onSubmit={handleLogin}>
                        <input
                            type="password"
                            className="w-full text-center text-xl font-bold py-4 rounded-xl border-2 border-slate-200 mb-4 focus:ring-4 focus:ring-indigo-100 outline-none"
                            placeholder="Enter Admin Password"
                            value={passwordInput}
                            onChange={e => setPasswordInput(e.target.value)}
                            autoFocus
                        />
                        <button className="w-full py-4 bg-slate-900 text-white font-bold rounded-xl hover:bg-black transition-all">Unlock Panel</button>
                    </form>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 w-screen h-screen bg-slate-50 overflow-auto grid place-items-center p-8">
            <div className="w-full max-w-4xl">
                <h1 className="text-3xl font-black text-slate-800 mb-6 flex items-center gap-3">
                    <Icon.Settings className="text-slate-400" />
                    Admin Panel: Access Codes
                </h1>

                <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100 mb-8">
                    <div className="flex gap-4 items-end mb-8">
                        <div>
                            <label className="block text-sm font-black text-slate-800 mb-2">Count</label>
                            <input
                                type="number"
                                value={genCount}
                                onChange={e => setGenCount(Number(e.target.value))}
                                className="w-24 px-4 py-3 rounded-xl bg-slate-50 border border-slate-200 font-bold"
                            />
                        </div>
                        <button
                            onClick={generateCodes}
                            disabled={loading}
                            className="px-8 py-3 bg-indigo-600 text-white rounded-xl font-bold shadow-lg hover:bg-indigo-700 active:scale-95 transition-all"
                        >
                            {loading ? "Generating..." : "Generate Codes"}
                        </button>

                        <button
                            onClick={() => fetchCodes(token)}
                            className="px-8 py-3 bg-slate-100 text-slate-600 rounded-xl font-bold hover:bg-slate-200 active:scale-95 transition-all ml-auto"
                        >
                            Refresh
                        </button>
                    </div>

                    <div className="overflow-hidden rounded-2xl border border-slate-200">
                        <table className="w-full text-left">
                            <thead className="bg-slate-50 text-slate-500 font-bold uppercase text-xs">
                                <tr>
                                    <th className="p-4">Code</th>
                                    <th className="p-4">Status</th>
                                    <th className="p-4">Copy Count</th>
                                    <th className="p-4">Created</th>
                                    <th className="p-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {codes.map((c) => (
                                    <tr key={c.code} className="hover:bg-slate-50 transition-colors">
                                        <td className="p-4 font-mono font-bold text-lg text-slate-800">{c.code}</td>
                                        <td className="p-4">
                                            {c.used ? (
                                                <span className="px-3 py-1 bg-red-100 text-red-700 rounded-lg text-xs font-bold">USED</span>
                                            ) : (
                                                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-lg text-xs font-bold">ACTIVE</span>
                                            )}
                                        </td>
                                        <td className="p-4 font-bold text-slate-600">{c.copy_count}</td>
                                        <td className="p-4 text-sm text-slate-400">{new Date(c.created_at).toLocaleString()}</td>
                                        <td className="p-4 text-right">
                                            <button
                                                onClick={() => handleCopy(c.code)}
                                                className="px-4 py-2 bg-white border border-slate-200 text-slate-600 rounded-lg text-sm font-bold hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 transition-all flex items-center gap-2 ml-auto"
                                            >
                                                <Icon.Copy size={16} /> Copy
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                                {codes.length === 0 && (
                                    <tr>
                                        <td colSpan={5} className="p-8 text-center text-slate-400 font-medium">No codes found. Generate some!</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

    );
};
