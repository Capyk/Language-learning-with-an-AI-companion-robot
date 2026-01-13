// src/config/api.ts

/**
 * Sztywny adres backendu.
 * Używa zmiennych środowiskowych Vite (import.meta.env) do wykrycia trybu developera.
 */
export const API_BASE = import.meta.env.DEV
  ? 'http://127.0.0.1:8000'
  : 'https://german-learning-language-backend.onrender.com';

