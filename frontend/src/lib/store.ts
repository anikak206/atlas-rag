import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  role: 'Admin' | 'Finance' | 'Intern';
  tenant_id: string;
  full_name?: string;
}

interface Tenant {
  id: string;
  name: string;
  slug: string;
}

interface AppState {
  user: User | null;
  tenant: Tenant | null;
  currentRole: 'Admin' | 'Finance' | 'Intern';
  currentTenant: string;
  setUser: (user: User | null) => void;
  setTenant: (tenant: Tenant | null) => void;
  setCurrentRole: (role: 'Admin' | 'Finance' | 'Intern') => void;
  setCurrentTenant: (tenant: string) => void;
  conversations: any[];
  setConversations: (convs: any[]) => void;
  activeConversationId: string | null;
  setActiveConversationId: (id: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  user: null,
  tenant: null,
  currentRole: 'Admin',
  currentTenant: 'acme',
  setUser: (user) => set({ user }),
  setTenant: (tenant) => set({ tenant }),
  setCurrentRole: (role) => set({ currentRole: role }),
  setCurrentTenant: (tenant) => set({ currentTenant: tenant }),
  conversations: [],
  setConversations: (conversations) => set({ conversations }),
  activeConversationId: null,
  setActiveConversationId: (id) => set({ activeConversationId: id }),
}));
