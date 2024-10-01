#!/usr/sbin/dtrace -s

/* Trace read operations */
syscall::read:entry
{
    self->trace_read = 1; /* Flag to indicate a read operation is in progress */
    self->raddr = arg1; /* Save the buffer address */
}

syscall::read:return
/self->trace_read/
{
    printf("%p READ %p\n", uregs[R_PC], self->raddr);
    self->trace_read = 0; /* Clear the flag */
}

/* Trace write operations */
syscall::write:entry
{
    self->trace_write = 1; /* Flag to indicate a write operation is in progress */
    self->waddr = arg1; /* Save the buffer address */
}

syscall::write:return
/self->trace_write/
{
    printf("%p WRITE %p\n", uregs[R_PC], self->waddr);
    self->trace_write = 0; /* Clear the flag */
}
