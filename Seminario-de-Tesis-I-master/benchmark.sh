#! /bin/bash

tiempo=1 #segundos
runtime="10 second"
endtime=$(date -ud "$runtime" +%s)

echo "instr,ciclos,instr/ciclo,e_cpus,e_ram,cpu_used,ram_used" >> ~/results/perf_SVC_exh.csv

while [[ $(date -u +%s) -le $endtime ]]
do
        
    #mhora=$(date +%H:%M:%S.%N)
        
    stats_inst_cycle=$((sudo perf stat -e cycles,instructions -x ' ' -ag sleep $tiempo) 2>&1 | cut -d ' ' -f1)
    cycles=$(echo $stats_inst_cycle | cut -d ' ' -f1)
    instructions=$(echo $stats_inst_cycle | cut -d ' ' -f2)
    instructions_per_cycle=$(($instructions/$cycles))

    stats_energy=$((sudo perf stat -e energy-cores,energy-ram -x ' ' -ag sleep $tiempo) 2>&1  | cut -d ' ' -f1)
    energy_cpus=$(echo $stats_energy | cut -d ' ' -f1)
    energy_ram=$(echo $stats_energy | cut -d ' ' -f2)

    cpu_used=$(top -bn 1 -d $tiempo | grep '^%Cpu' | tail -n 1 | gawk '{print $2+$4+$6}')
    mem_used=$(top -bn 1 -d $tiempo | grep 'MiB Mem' | tail -n 1 | gawk '{print $6}')

    echo "$instructions_per_cycle,$energy_cpus,$energy_ram,$cpu_used,$mem_used" >> ~/results/perf_SVC_exh.csv
    
done
