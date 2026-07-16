import { createClient } from '@supabase/supabase-js'
const supabase = createClient(process.env.URL, process.env.KEY)
export const up = (f) => supabase.storage.from('avatars').upload('x', f)
