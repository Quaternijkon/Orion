#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use std::sync::Barrier;
use std::time::Instant;
fn faults()->u64 {std::fs::read_to_string("/proc/self/stat").unwrap().split_whitespace().nth(9).unwrap().parse().unwrap()}
fn main(){
 for nodes in [31_250_usize,31_250_000] {
  for reuse in [false,true,true,false] {
   let gate=Barrier::new(17);let before=faults();let origin=Instant::now();
   let results=std::thread::scope(|scope|{
    let mut workers=Vec::new();
    for worker in 0..16 {
     let gate=&gate;
     workers.push(scope.spawn(move||{
      let mut words=if reuse {vec![0_usize;nodes.div_ceil(usize::BITS as usize)]}else{Vec::new()};
      let mut touched=Vec::new();let mut rng=worker as u64+1;let mut check=0;
      gate.wait();let begin=origin.elapsed();
      for _ in 0..250 {
       if reuse {for i in touched.drain(..){words[i]=0;}}
       else {words=vec![0_usize;nodes.div_ceil(usize::BITS as usize)];touched=Vec::new();}
       for _ in 0..4096 {
        rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i=(rng as usize)%nodes;let wi=i/(usize::BITS as usize);let bit=1_usize<<(i%(usize::BITS as usize));
        if words[wi]==0{touched.push(wi);}
        check+=usize::from(words[wi]&bit==0);words[wi]|=bit;
       }
       std::hint::black_box(&words);
       if !reuse {drop(std::mem::take(&mut words));drop(std::mem::take(&mut touched));}
      }
      (begin,origin.elapsed(),check)
     }));
    }
    gate.wait();workers.into_iter().map(|w|w.join().unwrap()).collect::<Vec<_>>()
   });
   let span=(results.iter().map(|r|r.1).max().unwrap()-results.iter().map(|r|r.0).min().unwrap()).as_secs_f64();
   println!("{}",serde_json::json!({"nodes":nodes,"reuse":reuse,"queries":4000,"synthetic_visits_per_query":4096,"bitmap_bytes":nodes.div_ceil(usize::BITS as usize)*8,"wall_seconds":span,"mean_thread_ms_per_query":results.iter().map(|r|(r.1-r.0).as_secs_f64()*1000.0).sum::<f64>()/4000.0,"minor_faults":faults()-before,"checksum":results.iter().map(|r|r.2).sum::<usize>(),"scope":"allocation/touch isolation only, not graph search"}));
  }
 }
}
