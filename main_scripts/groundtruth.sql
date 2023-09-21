create view cat_name as
select sid, category_key, category_name from verint.sessions_categories left join verint.categories using category_key
where category_name in ("Cable Customer Frustration",
            "L1R - Tech Issues: Internet",
            "BP1 Technical Support",
            "BP3 Technical Support",
            "HOT TOPIC: National Outage",
            "Ignite TV",
            "L1R - Tech Issues: TV",
            "L2R - INT: Modem Offline",
            "HOT TOPIC: Outage Compensation",
            "L1F - Tech Issues: Internet",
            "L2R - General Inquiries: Internet");

create view sessions_booked_cleaned as
select distinct CONCAT(unit_num, '0', channel_num) as speech_id_verint, sid
from verint.sessions_booked;

create view sum_fct_cleaned as
select distinct speech_id_verint, customer_id as account_number, conversation_date as event_date
from verint.cbu_rog_conversation_sumfct;

create view inter_table  as
select * from sum_fct_cleaned join sessions_booked_cleaned using speech_id_verint;

create table ml_etl_output.gt_call_prediction as
select account_number, event_date,category_name from inter_table join cat_name using sid;


