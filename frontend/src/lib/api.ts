import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_PREFIX = process.env.NEXT_PUBLIC_API_PREFIX || '/api/v1';

export const api = axios.create({
  baseURL: `${API_URL}${API_PREFIX}`,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    const tenantId = localStorage.getItem('tenant_id');
    if (tenantId) {
      config.headers['X-Tenant-ID'] = tenantId;
    }
  }
  return config;
});

export const setAuthToken = (token: string) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('access_token', token);
  }
};

export const clearAuth = () => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    localStorage.removeItem('tenant_id');
  }
};
