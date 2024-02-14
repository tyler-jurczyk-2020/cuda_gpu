--FIRST LAYER
--Create a view with all kernel calls and cudamemcpys and cudamemcpystosymbol sorted by start time
create view if not exists calls as select  * from (select value, start, end, type from (select *, "api" as type from CUPTI_ACTIVITY_KIND_MEMCPY left join CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_MEMCPY.correlationId = CUPTI_ACTIVITY_KIND_RUNTIME.correlationId) LEFT JOIN StringIds where nameId = id UNION select value, start, end, "kernel" as type from CUPTI_ACTIVITY_KIND_KERNEL left join StringIds on shortname = id UNION select value, start, end, "api" as type from CUPTI_ACTIVITY_KIND_RUNTIME LEFT JOIN StringIds where nameId = id and value like "cudaMemcpyToSymbol%" UNION select value, start, end, type from (select *, "api" as type from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION left join CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.correlationId = CUPTI_ACTIVITY_KIND_RUNTIME.correlationId) LEFT JOIN StringIds where nameId = id) ORDER BY start;
--Create a view with all stuff from previous view with a smaller end time than the first "do_not_remove_this_kernel" kernel
create view if not exists lesscalls as select * from calls where end < (select start from calls where value == "do_not_remove_this_kernel" LIMIT 1);
--Query the sum of all GPU kernel times in the previous view not including the "prefn_marker_kernel"
select 'Layer 1 GPUTime: ' || IFNULL((sum(end-start)*1.0/1000000), 0) || ' ms' from calls where end < (select start from calls where value == "do_not_remove_this_kernel" LIMIT 1) and type == "kernel" and value != "prefn_marker_kernel";
--Query in the lesscalls view the difference between the start time of the last API call that isn't "cudaDeviceSynchronize" before the "do_not_remove_this_kernel" kernel and the end time of the last API call before the first non-"do_not_remove_this_kernel" kernel that isn't the "prefn_marker_kernel"
select 'Layer 1 OpTime: ' || IFNULL((((select start from lesscalls ORDER BY start DESC LIMIT 1) - (select end from lesscalls where type == "api" and value not like "cudaDeviceSynchronize%" and end < (select start from lesscalls where type == "kernel" and value != "prefn_marker_kernel" LIMIT 1) ORDER BY start DESC LIMIT 1))*1.0/1000000), 0) || ' ms';
--Query all time between end of "cudaDeviceSynchronzie" before the "prefn_marker_kernel" kernel and start of "do_not_remove_this_kernel"
select 'Layer 1 LayerTime: ' || IFNULL(((select start from calls where value == "do_not_remove_this_kernel" LIMIT 1)-(select end from lesscalls where value like "cudaDeviceSynchronize%" LIMIT 1))*1.0/1000000, 0) || ' ms';

--SECOND LAYER
--Create a view with all kernel calls and cudamemcpys and cudamemcpystosymbol after the second "prefn_marker_kernel" kernel sorted by start time
create view if not exists callsnew as select * from calls where start > (select end from calls where value == "prefn_marker_kernel" ORDER BY start DESC LIMIT 1);
--Create a view with all stuff from previous view with a smaller end time than the first "do_not_remove_this_kernel" kernel
create view if not exists lesscallsnew as select * from callsnew where end < (select start from callsnew where value == "do_not_remove_this_kernel" LIMIT 1);
--Query the sum of all GPU kernel times in the previous view not including the "prefn_marker_kernel"
select 'Layer 2 GPUTime: ' || IFNULL((sum(end-start)*1.0/1000000), 0) || ' ms' from callsnew where end < (select start from callsnew where value == "do_not_remove_this_kernel" LIMIT 1) and type == "kernel"  and value != "prefn_marker_kernel";
--Query in the lesscallsnew view the difference between the start time of the last API call that isn't "cudaDeviceSynchronize" before the "do_not_remove_this_kernel" kernel and the end time of the last API call before the first non-"do_not_remove_this_kernel" kernel that isn't the "prefn_marker_kernel"
select 'Layer 2 OpTime: ' || IFNULL((((select start from lesscallsnew ORDER BY start DESC LIMIT 1) - (select end from lesscallsnew where type == "api" and value not like "cudaDeviceSynchronize%" and end < (select start from lesscallsnew where type == "kernel" and value != "prefn_marker_kernel" LIMIT 1) ORDER BY start DESC LIMIT 1))*1.0/1000000), 0) || ' ms'; 
--Query all time between end of "cudaDeviceSynchronzie" before the "prefn_marker_kernel" kernel and start of "do_not_remove_this_kernel"
select 'Layer 2 LayerTime: ' || IFNULL(((select start from callsnew where value == "do_not_remove_this_kernel" LIMIT 1)-(select end from lesscallsnew where value like "cudaDeviceSynchronize%" LIMIT 1))*1.0/1000000, 0) || ' ms';

drop view calls;
drop view callsnew;
drop view lesscalls;
drop view lesscallsnew;
