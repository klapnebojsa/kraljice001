(ns kraljice001.core
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release]]            
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer :all]
             [legacy :refer :all]
             [constants :refer :all]
             [toolbox :refer :all]
             [utils :refer :all]]
            [vertigo
             [bytes :refer [buffer direct-buffer byte-seq byte-count slice]]
             [structs :refer [int8 int32 int64 wrap-byte-seq]]])
  (:import [org.jocl CL]))

(set! *unchecked-math* true)
(require 'clojure.edn)
;(try 
 (with-release [;dev (nth  (sort-by-cl-version (devices (first (platforms)))) 0)
                platformsone (first (platforms))
                dev (nth  (sort-by-cl-version (devices platformsone)) 0)
                ;dev (nth  (sort-by-cl-version (devices platformsone)) 1)
                ctx (context [dev])
                cqueue (command-queue-1 ctx dev :profiling)]
  
 (println (vendor platformsone))
 (println "dev: " (name-info dev))
 (facts
   "Listing on page 225."
   (let [program-source
         (slurp (io/reader "examples/kraljice.cl"))
         ;num-items (Math/pow 2 10)                    ;2 na 20-tu = 1048576
         num-items 88888888
         bytesize (* num-items Float/BYTES)           ;Float/BYTES = 4    =>   bytesize = 4 * 2na20 = 4 * 1048576 = 4194304.0
         workgroup-size 256
         notifications (chan)
         follow (register notifications)
         brpolja 6
         data #_(int-array                  ;deo za formiranje ulaznih parametara za problem 8 dama
                  (with-local-vars [p ()]
                    (let [k (atom 0) n brpolja x (make-array Integer/TYPE n)]
                      (while (or (pos? @k) (zero? @k))
                        (aset x @k (inc (aget x @k)))
                        (if-not (> (aget x @k) n)
                          (do                            ;then 1
                            (if (= @k (- n 1))
                              (var-set p (conj @p (Integer. (clojure.string/join "" (vec x))))) ;then 2
                              (aset x (swap! k inc) 0)))                                        ;else 2
                          (swap! k dec))))               ;else 1
                    @p))
         (int-array (clojure.edn/read-string (slurp "podaci/kraljice.dat")))     ;deo za citanje ulaznih podataka za problem 8 dama
         
         ;data (float-array (repeatedly num-items #(rand-int num-items)))
         ;data (float-array (repeatedly num-items #(let [p (rand-int num-items)]
         ;                                           (println p)
         ;                                           p)))
       
         cl-partial-sums (* workgroup-size Float/BYTES)       ;4 * 256 = 1024
         partial-output (float-array (/ bytesize workgroup-size))      ;4*2na20 / 256 = 4*2na20 / 2na8 = 4*2na12   - niz od 16384 elemenata
         output (float-array 1)               ;pocetna vrednost jedan clan sa vrednoscu 0.0
         ] 
     
         ;(spit "podaci/kraljice.dat" (prn-str (seq data)))   ;Upisuje podatke u fajl kraljice.dat    
         (println "data" (seq data))
        
     (println "ooooooooooooooooo")
     (with-release [cl-data (cl-buffer ctx bytesize :read-only)
                    cl-brpolja (cl-buffer ctx bytesize :read-only)                    
                    cl-output (cl-buffer ctx bytesize :write-only)
                    cl-partial-output (cl-buffer ctx (/ bytesize workgroup-size)   ;kreira cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
                                       :read-write)
                    prog (build-program! (program-with-source ctx [program-source]))   ;kreira program u kontekstu ctx sa kodom programa u kojem se nalaze tri kernela 
                    kraljice_brojac (kernel prog "kraljice_brojac")            ;definise kernel iz prog
                    ;reduction-scalar (kernel prog "reduction_scalar")          ;definise kernel iz prog
                    ;reduction-vector (kernel prog "reduction_vector")          ;definise kernel iz prog
                    ;reduction-complete (kernel prog "reduction_complete")      ;definise kernel iz prog                    
                    profile-event (event)                  ;kreira novi cl_event (dogadjaj)
                    ;profile-event1 (event)                 ;          -||-
                    ;profile-event2 (event)                 ;          -||- 
                    ;profile-event3 (event)
                    ]                ;          -||-       
       ;(println "(apply + (float-array (range 0" num-items "))): " (apply + data))
 
       (facts
         
       (println "============ Naive reduction ======================================")
       
        ;; ============ Naive reduction ======================================
        (set-args! kraljice_brojac cl-data cl-brpolja cl-partial-sums cl-output) => kraljice_brojac
        ;(set-args! kraljice_brojac cl-data cl-output) => kraljice_brojac
        (enq-write! cqueue cl-data data) => cqueue                                 ;SETUJE VREDNOST GLOBALNE PROMENJIVE cl-data SA VREDNOSCU data
        ;(enq-write! cqueue cl-brpolja brpolja) => cqueue
        
        ;(println "data: " (seq data))        
        (enq-nd! cqueue kraljice_brojac (work-size [1]) nil profile-event)
        => cqueue
        (follow profile-event) => notifications
        (enq-read! cqueue cl-output output) => cqueue
        (finish! cqueue) => cqueue
        (println "Naive reduction time:"
                 (-> (<!! notifications) :event profiling-info durations :end))
        (println "Naive output: " (seq output))
        ;(println "sta je data: " data)        
        ;(aget output 0) => num-items
 
        )))))
 
 
 