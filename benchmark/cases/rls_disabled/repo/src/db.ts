import { createClient } from '@supabase/supabase-js'
const supabase = createClient(process.env.URL, process.env.KEY)
export const orders = () => supabase.from('orders')
