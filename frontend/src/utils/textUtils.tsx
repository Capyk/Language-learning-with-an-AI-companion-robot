// src/utils/textUtils.tsx
import React from 'react';

/**
 * Formatuje tekst parsując proste znaczniki pogrubienia.
 * Zamienia "**tekst**" na: <strong class="...">tekst</strong>
 * * @param text - Tekst wejściowy (może być null/undefined)
 * @returns Tablica elementów React lub null
 */
export const formatText = (text: string | null | undefined): React.ReactNode => {
    if (!text) return null;

    // Dzieli tekst zachowując dopasowania w nawiasach (czyli nasze pogrubienia)
    const parts = text.split(/(\*\*.*?\*\*)/g);

    return parts.map((part, index) => {
        // Sprawdzamy czy fragment jest objęty gwiazdkami
        if (part.startsWith('**') && part.endsWith('**')) {
            return (
                <strong key={index} className="font-black text-indigo-900">
                    {part.slice(2, -2)}
                </strong>
            );
        }
        // Zwykły tekst
        return <span key={index}>{part}</span>;
    });
};